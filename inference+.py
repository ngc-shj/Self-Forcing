"""
Improved Inference for Self-Forcing.
Combines functionality from demo.py and inference.py for enhanced video generation.
"""

import os
import re
import random
import time
import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


def parse_arguments():
    """Parse command line arguments combining demo.py and inference.py options."""
    parser = argparse.ArgumentParser(description="Self-Forcing Improved Inference")
    
    # Basic configuration
    parser.add_argument("--config_path", type=str, default='./configs/self_forcing_dmd.yaml',
                        help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/self_forcing_dmd.pt',
                        help="Path to the checkpoint folder")
    
    # Data and output
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
    parser.add_argument("--output_folder", type=str, default="./output_videos",
                        help="Output folder for generated videos")
    parser.add_argument("--image_output_folder", type=str, default="./output_images",
                        help="Output folder for frame images")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, help="Single prompt for generation")
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Number of frames to generate")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate per prompt")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--fps", type=int, default=6, help="Output video framerate")
    
    # Model options
    parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--use_taehv", action="store_true", help="Use TAEHV VAE for faster decoding")
    parser.add_argument("--enable_torch_compile", action="store_true", help="Enable torch.compile for speedup")
    parser.add_argument("--enable_fp8", action="store_true", help="Enable FP8 quantization")
    parser.add_argument("--trt", action="store_true", help="Use TensorRT for VAE")
    
    # Output options
    parser.add_argument("--save_frames", action="store_true", help="Save individual frames as images")
    parser.add_argument("--save_with_index", action="store_true",
                        help="Whether to save the video using the index or prompt as the filename")
    parser.add_argument("--no_video", action="store_true", help="Don't save video files, only frames")
    
    # Distributed training
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed inference")
    
    return parser.parse_args()


def initialize_distributed():
    """Initialize distributed inference if needed."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda")
        local_rank = 0
        world_size = 1
    
    return device, local_rank, world_size


def initialize_vae_decoder(use_taehv=False, use_trt=False):
    """Initialize VAE decoder based on the selected option."""
    if use_trt:
        from demo_utils.vae import VAETRTWrapper
        return VAETRTWrapper()

    if use_taehv:
        from demo_utils.taehv import TAEHV
        # Check if taew2_1.pth exists in checkpoints folder, download if missing
        taehv_checkpoint_path = "checkpoints/taew2_1.pth"
        if not os.path.exists(taehv_checkpoint_path):
            print(f"taew2_1.pth not found in checkpoints folder {taehv_checkpoint_path}. Downloading...")
            os.makedirs("checkpoints", exist_ok=True)
            download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            try:
                urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
                print(f"Successfully downloaded taew2_1.pth to {taehv_checkpoint_path}")
            except Exception as e:
                print(f"Failed to download taew2_1.pth: {e}")
                raise

        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        class TAEHVDiffusersWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float16
                self.taehv = TAEHV(checkpoint_path=taehv_checkpoint_path).to(self.dtype)
                self.config = DotDict(scaling_factor=1.0)

            def decode(self, latents, return_dict=None):
                return self.taehv.decode_video(latents, parallel=False).mul_(2).sub_(1)

        return TAEHVDiffusersWrapper()
    else:
        vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        vae_decoder.load_state_dict(decoder_state_dict)
        return vae_decoder


def setup_models(args, device):
    """Setup and load all required models."""
    print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    
    # Load config
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    
    # Initialize models
    text_encoder = WanTextEncoder()
    
    # Initialize VAE decoder
    vae_decoder = initialize_vae_decoder(use_taehv=args.use_taehv, use_trt=args.trt)
    vae_decoder.eval()
    vae_decoder.to(dtype=torch.float16)
    vae_decoder.requires_grad_(False)
    vae_decoder.to(device)
    
    # Initialize transformer
    transformer = WanDiffusionWrapper(is_causal=True)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        #transformer.load_state_dict(state_dict['generator_ema' if args.use_ema else 'generator'])
        transformer.load_state_dict(state_dict['generator_ema'])
    
    text_encoder.eval()
    transformer.eval()
    transformer.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.bfloat16)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # Initialize pipeline
    if hasattr(config, 'denoising_step_list'):
        pipeline = CausalInferencePipeline(config, device=device, generator=transformer, 
                                         text_encoder=text_encoder, vae=vae_decoder)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)
        pipeline.generator = transformer
        pipeline.text_encoder = text_encoder
        pipeline.vae = vae_decoder
    
    # Apply optimizations
    if args.enable_fp8:
        print("🔧 Applying FP8 quantization to transformer")
        from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
        quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
    
    if args.enable_torch_compile:
        print("🔥 Compiling models with torch.compile")
        transformer.compile(mode="max-autotune-no-cudagraphs")
        if not args.use_taehv and not low_memory and not args.trt:
            vae_decoder.compile(mode="max-autotune-no-cudagraphs")
    
    # Move models to device
    if low_memory:
        DynamicSwapInstaller.install_model(text_encoder, device=device)
    else:
        text_encoder.to(device)
    transformer.to(device)
    
    return pipeline, config, low_memory


def tensor_to_image(frame_tensor, output_path):
    """Convert a single frame tensor to image and save."""
    # Clamp and normalize to 0-255
    frame = torch.clamp(frame_tensor.float(), -1., 1.) * 127.5 + 127.5
    frame = frame.to(torch.uint8).cpu().numpy()
    
    # CHW -> HWC
    if len(frame.shape) == 3:
        frame = np.transpose(frame, (1, 2, 0))
    
    # Convert to PIL Image and save
    if frame.shape[2] == 3:  # RGB
        image = Image.fromarray(frame, 'RGB')
    else:  # Handle other formats
        image = Image.fromarray(frame)
    
    image.save(output_path, format='JPEG', quality=95)


def generate_mp4_from_images(image_directory, output_video_path, fps=24):
    """Generate an MP4 video from a directory of images."""
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite existing files
        '-framerate', str(fps),
        '-i', os.path.join(image_directory, '%03d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Video saved to {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        print(f"FFmpeg stdout: {e.stdout.decode()}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")


def calculate_sha256(data):
    """Calculate SHA-256 hash of data."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def create_output_name(prompt, seed, idx=None):
    """Create a unique output name based on prompt and seed."""
    # Extract words up to the first punctuation or newline
    words_up_to_punctuation = re.split(r'[^\w\s]', prompt)[0].strip() if prompt else ''
    if not words_up_to_punctuation:
        words_up_to_punctuation = re.split(r'[\n\r]', prompt)[0].strip()
    
    # Calculate SHA-256 hash of the entire prompt
    sha256_hash = calculate_sha256(prompt)
    
    # Create name with the extracted words and hash
    if idx is not None:
        return f"{idx:04d}_{words_up_to_punctuation[:20]}_{seed}_{sha256_hash[:10]}"
    else:
        return f"{words_up_to_punctuation[:20]}_{seed}_{sha256_hash[:10]}"


@torch.no_grad()
def generate_video_advanced(pipeline, args, prompt, seed, idx=None, low_memory=False):
    """Advanced video generation with frame saving and multiple options."""
    device = pipeline.device if hasattr(pipeline, 'device') else gpu
    
    print(f"🎬 Generating video for prompt: '{prompt[:50]}...'")
    generation_start_time = time.time()
    
    # Create output name
    output_name = create_output_name(prompt, seed, idx)
    
    # Setup output directories
    if args.save_frames:
        frame_dir = os.path.join(args.image_output_folder, output_name)
        os.makedirs(frame_dir, exist_ok=True)
    
    # Text encoding
    print("🔤 Encoding text prompt...")
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
    for key, value in conditional_dict.items():
        conditional_dict[key] = value.to(dtype=torch.float16)
    
    if low_memory:
        gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
        move_model_to_device_with_memory_preservation(
            pipeline.text_encoder, target_device=device, preserved_memory_gb=gpu_memory_preservation)
    
    # Initialize generation
    print("🚀 Initializing generation...")
    rnd = torch.Generator(device).manual_seed(seed)
    
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=device)
    
    noise = torch.randn([1, 21, 16, 60, 104], device=device, dtype=torch.float16, generator=rnd)
    
    # Generation parameters
    num_blocks = 7
    current_start_frame = 0
    num_input_frames = 0
    all_num_frames = [pipeline.num_frame_per_block] * num_blocks
    
    # Initialize VAE cache
    if args.use_taehv:
        vae_cache = None
    else:
        vae_cache = ZERO_VAE_CACHE
        for i in range(len(vae_cache)):
            vae_cache[i] = vae_cache[i].to(device=device, dtype=torch.float16)
    
    all_frames = []
    total_frames_generated = 0
    
    # Generate blocks
    for idx_block, current_num_frames in enumerate(tqdm(all_num_frames, desc="Generating blocks")):
        print(f"🔄 Processing block {idx_block+1}/{len(all_num_frames)}")
        
        noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
        
        # Denoising loop
        for index, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones([1, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
            
            if index < len(pipeline.denoising_step_list) - 1:
                _, denoised_pred = pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length
                )
                next_timestep = pipeline.denoising_step_list[index + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([1 * current_num_frames], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length
                )
        
        # Update KV cache for next block
        if idx_block != len(all_num_frames) - 1:
            pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=torch.zeros_like(timestep),
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
        
        # Decode to pixels
        print(f"🎨 Decoding block {idx_block+1} to pixels...")
        if args.trt:
            all_current_pixels = []
            for i in range(denoised_pred.shape[1]):
                is_first_frame = torch.tensor(1.0).cuda().half() if idx_block == 0 and i == 0 else torch.tensor(0.0).cuda().half()
                outputs = pipeline.vae.forward(denoised_pred[:, i:i + 1, :, :, :].half(), is_first_frame, *vae_cache)
                current_pixels, vae_cache = outputs[0], outputs[1:]
                all_current_pixels.append(current_pixels.clone())
            pixels = torch.cat(all_current_pixels, dim=1)
            if idx_block == 0:
                pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
        else:
            if args.use_taehv:
                if vae_cache is None:
                    vae_cache = denoised_pred
                else:
                    denoised_pred = torch.cat([vae_cache, denoised_pred], dim=1)
                    vae_cache = denoised_pred[:, -3:, :, :, :]
                pixels = pipeline.vae.decode(denoised_pred)
                if idx_block == 0:
                    pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
                else:
                    pixels = pixels[:, 12:, :, :, :]
            else:
                pixels, vae_cache = pipeline.vae(denoised_pred.half(), *vae_cache)
                if idx_block == 0:
                    pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
        
        # Save frames if requested
        if args.save_frames:
            block_frames = pixels.shape[1]
            for frame_idx in range(block_frames):
                frame_tensor = pixels[0, frame_idx].cpu()
                frame_path = os.path.join(frame_dir, f"{total_frames_generated+frame_idx:03d}.jpg")
                tensor_to_image(frame_tensor, frame_path)
        
        # Collect frames for video
        all_frames.append(pixels.cpu())
        total_frames_generated += pixels.shape[1]
        current_start_frame += current_num_frames
    
    # Combine all frames
    video_tensor = torch.cat(all_frames, dim=1)
    video_tensor = rearrange(video_tensor, 'b t c h w -> b t h w c')
    video_tensor = (torch.clamp(video_tensor, -1., 1.) + 1.) * 127.5
    video_tensor = video_tensor.to(torch.uint8)
    
    generation_time = time.time() - generation_start_time
    print(f"🎉 Generation completed in {generation_time:.2f}s! Generated {total_frames_generated} frames")
    
    # Save video
    if not args.no_video:
        video_path = os.path.join(args.output_folder, f"{output_name}.mp4")
        write_video(video_path, video_tensor[0], fps=args.fps)
        print(f"✅ Video saved to {video_path}")
        
        # Alternative: create video from saved frames if available
        if args.save_frames:
            alt_video_path = os.path.join(args.output_folder, f"{output_name}_from_frames.mp4")
            generate_mp4_from_images(frame_dir, alt_video_path, args.fps)
    
    return video_tensor, output_name, generation_time


def main():
    args = parse_arguments()
    
    # Initialize distributed setup
    device, local_rank, world_size = initialize_distributed()
    
    # Set seed
    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
    
    set_seed(args.seed + local_rank)
    torch.set_grad_enabled(False)
    
    # Create output directories
    os.makedirs(args.output_folder, exist_ok=True)
    if args.save_frames:
        os.makedirs(args.image_output_folder, exist_ok=True)
    
    # Setup models
    pipeline, config, low_memory = setup_models(args, device)
    
    # Single prompt generation
    if args.prompt:
        print(f"🎯 Single prompt generation mode")
        for sample_idx in range(args.num_samples):
            current_seed = args.seed + sample_idx
            video, output_name, gen_time = generate_video_advanced(
                pipeline, args, args.prompt, current_seed, low_memory=low_memory
            )
            print(f"✅ Sample {sample_idx+1}/{args.num_samples} completed: {output_name}")
        return
    
    # Batch processing from dataset
    if not args.data_path:
        print("❌ Error: Either --prompt or --data_path must be provided")
        return
    
    print(f"📚 Batch processing mode from {args.data_path}")
    
    # Create dataset
    if args.i2v:
        assert not dist.is_initialized(), "I2V does not support distributed inference yet"
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = TextImagePairDataset(args.data_path, transform=transform)
    else:
        dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
    
    num_prompts = len(dataset)
    print(f"📊 Number of prompts: {num_prompts}")
    
    # Setup dataloader
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)
    
    if dist.is_initialized():
        dist.barrier()
    
    # Process batches
    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0), desc="Processing prompts"):
        idx = batch_data['idx'].item()
        prompt = batch_data['prompts'][0]
        
        if idx >= num_prompts:
            continue
        
        print(f"\n📝 Processing prompt {idx}: {prompt[:100]}...")
        
        for sample_idx in range(args.num_samples):
            current_seed = args.seed + idx * args.num_samples + sample_idx
            
            try:
                video, output_name, gen_time = generate_video_advanced(
                    pipeline, args, prompt, current_seed, idx=idx, low_memory=low_memory
                )
                print(f"✅ Completed {output_name} in {gen_time:.2f}s")
            except Exception as e:
                print(f"❌ Failed to generate video for prompt {idx}: {e}")
                continue
    
    print("🎊 All generations completed!")


if __name__ == "__main__":
    main()
