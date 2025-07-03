#!/usr/bin/env python3
import argparse
import os
import sys
import random
import time
import json
import re

import torch
import numpy as np
from PIL import Image
import einops
from omegaconf import OmegaConf

# DDP imports
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast # For FP16


# --- Path Configuration ---
# Ensure the script can find modules in the ControlNet-main directory
CONTROLNET_MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'ControlNet-main'))
if CONTROLNET_MAIN_PATH not in sys.path:
    sys.path.append(CONTROLNET_MAIN_PATH)
LDM_PATH = os.path.join(CONTROLNET_MAIN_PATH, 'ldm')
if LDM_PATH not in sys.path:
    sys.path.append(LDM_PATH)

from model.creat_model import create_model, load_state_dict
from model.ddim_hacked import DDIMSampler
from model.hack import enable_sliced_attention # For memory saving


def setup_ddp(rank, world_size, port="12355"):
    """Initializes the DDP environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # Set device for this process

def cleanup_ddp():
    """Cleans up the DDP environment."""
    dist.destroy_process_group()

def seed_everything(seed, rank=0):
    """Set seed for reproducibility, with an offset for DDP ranks."""
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)

def preprocess_segmentation_map(image_path, target_canvas_resolution):
    """
    Loads and preprocesses a segmentation map.
    The long edge of the image is scaled to target_canvas_resolution, and the short edge is scaled proportionally.
    The content is then centered on a target_canvas_resolution x target_canvas_resolution canvas.
    Returns the canvas tensor, crop box coordinates, and content dimensions.
    """
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: Could not load image {image_path}. Error: {e}")
        raise

    original_w, original_h = pil_image.size
    if original_w == 0 or original_h == 0:
        raise ValueError(f"Original image dimensions ({original_w}x{original_h}) cannot be zero.")

    # Determine the scaled content dimensions (content_w, content_h)
    if original_w > original_h:
        content_w = target_canvas_resolution
        scale = target_canvas_resolution / original_w
        content_h = int(round(original_h * scale))
    else:
        content_h = target_canvas_resolution
        scale = target_canvas_resolution / original_h
        content_w = int(round(original_w * scale))
    
    content_w = max(1, content_w) # Ensure dimensions are at least 1
    content_h = max(1, content_h)

    img_np = np.array(pil_image).astype(np.uint8)

    import cv2
    # Resize the original image content to content_w, content_h
    # cv2.resize's dsize parameter is (width, height)
    resized_content_np = cv2.resize(
        img_np,
        (content_w, content_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4 # Use INTER_AREA for shrinking, LANCZOS4 for enlarging
    )

    # Create a square canvas of target_canvas_resolution x target_canvas_resolution
    canvas_np = np.zeros((target_canvas_resolution, target_canvas_resolution, 3), dtype=np.uint8)
    
    # Calculate padding to center the content
    pad_x = (target_canvas_resolution - content_w) // 2
    pad_y = (target_canvas_resolution - content_h) // 2

    # Paste the scaled content onto the center of the canvas
    # Determine the actual end coordinates for pasting to ensure content is within the canvas
    paste_end_y = pad_y + content_h
    paste_end_x = pad_x + content_w
    
    canvas_np[pad_y:paste_end_y, pad_x:paste_end_x, :] = resized_content_np
    
    # Normalize and convert to tensor
    normalized_canvas = (canvas_np.astype(np.float32) / 127.5) - 1.0
    control_tensor = torch.from_numpy(normalized_canvas).unsqueeze(0)
    control_tensor = einops.rearrange(control_tensor, 'b h w c -> b c h w').clone()
    
    # Coordinates for cropping the content from the generated (target_canvas_resolution x target_canvas_resolution) image
    # (x_start, y_start, x_end, y_end)
    crop_box_on_canvas = (pad_x, pad_y, paste_end_x, paste_end_y)
    
    # The actual dimensions of the content pasted onto the canvas
    final_content_dims = (content_w, content_h)

    return control_tensor, crop_box_on_canvas, final_content_dims

def ddp_worker(rank, world_size, args, actual_gpu_ids):
    """DDP worker function, executed by each GPU process."""
    worker_start_time = time.time() # Start time for the entire worker
    actual_gpu_id = actual_gpu_ids[rank]
    
    print(f"[Rank {rank}/GPU {actual_gpu_id}] Initializing DDP worker...")
    setup_ddp(rank, world_size, port=args.ddp_port)
    
    if args.seed != -1:
        seed_everything(args.seed, rank)
    else: 
        seed_everything(random.randint(0, 2**32 - 1) + rank, 0)

    # 1. Load Model (once per worker)
    load_model_start_time = time.time()
    if rank == 0: print(f"Loading model config: {args.config_path}")
    model = create_model(config_path=args.config_path)
    if rank == 0: print(f"Loading model weights: {args.decoder_model_path}")
    model.load_state_dict(load_state_dict(args.decoder_model_path, location='cpu'), strict=False)
    model = model.to(actual_gpu_id)

    if args.use_fp16:
        if rank == 0: print(f"[Rank {rank}] Using FP16 mode, converting model to half precision.")
        model = model.half()
    
    model_ddp = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    ddim_sampler = DDIMSampler(model_ddp.module)
    model_load_duration = time.time() - load_model_start_time
    if rank == 0:
        print(f"Model loading finished for all workers. Rank 0 took: {model_load_duration:.2f} seconds.")

    # 2. Distribute files and chunk into batches
    files_for_this_gpu = args.input_files[rank::world_size]
    if not files_for_this_gpu:
        print(f"[Rank {rank}/GPU {actual_gpu_id}] No images assigned to this worker, exiting.")
        cleanup_ddp()
        return
        
    file_batches = [
        files_for_this_gpu[i:i + args.batch_size] 
        for i in range(0, len(files_for_this_gpu), args.batch_size)
    ]
    print(f"[Rank {rank}/GPU {actual_gpu_id}] Assigned {len(files_for_this_gpu)} images, split into {len(file_batches)} batches (max {args.batch_size} each).")

    total_image_processing_time = 0
    
    for batch_idx, image_path_batch in enumerate(file_batches):
        batch_process_start_time = time.time()
        print(f"--- [Rank {rank}] Processing batch {batch_idx + 1}/{len(file_batches)} ({len(image_path_batch)} images) ---")

        # --- Prepare all inputs for the current batch ---
        batch_controls = []
        batch_prompts = []
        batch_crop_boxes = []
        batch_base_names = []
        batch_exts = []

        for input_image_path in image_path_batch:
            base_name_with_ext = os.path.basename(input_image_path)
            base_name, ext = os.path.splitext(base_name_with_ext)

            # --- Determine the prompt for the current image ---
            current_prompt = None
            if args.prompt_data:
                prompt_info = None
                # Compatible with JSON format generated by new encoder.py
                # 1. Extract ID ('2') from filename (e.g., '2_output_image.png')
                match = re.match(r'^(\d+)', base_name_with_ext)
                if match:
                    image_id = match.group(1)
                    # 2. Construct possible original filename prefix (e.g., '2_image.')
                    key_prefix = f"{image_id}_image."
                    found_key = None
                    # 3. Iterate through JSON keys to find a match for the prefix (e.g., '2_image.JPEG')
                    for key in args.prompt_data.keys():
                        if key.startswith(key_prefix):
                            found_key = key
                            break
                    if found_key:
                        prompt_info = args.prompt_data[found_key]

                # 4. If not found, try using the full filename as the key (for backward compatibility)
                if prompt_info is None and base_name_with_ext in args.prompt_data:
                    prompt_info = args.prompt_data[base_name_with_ext]

                if prompt_info:
                    # Prefer the subject/background structure
                    subject = prompt_info.get('subject')
                    background = prompt_info.get('background')
                    
                    if subject is not None and background is not None:
                        prompt_parts = [p for p in [subject, background] if p and p.strip()]
                        current_prompt = ", ".join(prompt_parts)
                    else:
                        # Fallback to reading the 'prompt' field directly
                        current_prompt = prompt_info.get('prompt')

            if not current_prompt:
                current_prompt = args.prompt

            if not current_prompt:
                print(f"Error: [Rank {rank}] - No prompt available for image '{base_name_with_ext}'. Skipping in this batch.")
                continue
            
            # --- Preprocess Image ---
            try:
                control_image_tensor, crop_box, content_dims = preprocess_segmentation_map(
                    input_image_path, args.image_resolution
                )
            except Exception as e:
                print(f"Error: [Rank {rank}] failed to preprocess image {input_image_path}: {e}. Skipping in this batch.")
                continue

            batch_controls.append(control_image_tensor)
            batch_prompts.append(current_prompt)
            batch_crop_boxes.append(crop_box)
            batch_base_names.append(base_name)
            batch_exts.append(ext)
            if rank == 0 and batch_idx == 0:
                print(f"  Image {base_name_with_ext} (size {content_dims[0]}x{content_dims[1]}) added to batch.")
        
        if not batch_controls:
            print(f"--- [Rank {rank}] Batch {batch_idx + 1} is empty (all images failed preprocessing), skipping. ---\n")
            continue
        
        # --- Consolidate batch data and run inference ---
        num_unique_images_in_batch = len(batch_controls)
        total_generations_this_batch = num_unique_images_in_batch * args.num_samples
        
        control_tensors = torch.cat(batch_controls, dim=0)
        del batch_controls
        
        if args.num_samples > 1:
            control_tensors = control_tensors.repeat_interleave(args.num_samples, dim=0)
            final_prompts = []
            for p in batch_prompts:
                final_prompts.extend([p] * args.num_samples)
        else:
            final_prompts = batch_prompts
        
        control_this_gpu = control_tensors.to(actual_gpu_id)
        del control_tensors
        if args.low_vram: torch.cuda.empty_cache()
        
        H, W = args.image_resolution, args.image_resolution

        if args.low_vram:
            model_ddp.module.low_vram_shift(device=actual_gpu_id, is_diffusing=False)

        full_prompts = [p + ', ' + args.a_prompt for p in final_prompts]
        cond_prompts_list = full_prompts
        neg_prompts_list = [args.n_prompt] * len(full_prompts)

        with autocast(enabled=args.use_fp16):
            c_crossattn = model_ddp.module.get_learned_conditioning(cond_prompts_list)
            uc_crossattn = model_ddp.module.get_learned_conditioning(neg_prompts_list)
        
        if args.low_vram:
            model_ddp.module.low_vram_shift(device='cpu', is_diffusing=False)
            torch.cuda.empty_cache()

        cond = {"c_concat": [control_this_gpu], "c_crossattn": [c_crossattn]}
        un_cond = {"c_concat": None if args.guess_mode else [control_this_gpu], "c_crossattn": [uc_crossattn]}
        del final_prompts, cond_prompts_list, neg_prompts_list, c_crossattn, uc_crossattn
        if args.low_vram: torch.cuda.empty_cache()

        if args.low_vram:
            model_ddp.module.low_vram_shift(device=actual_gpu_id, is_diffusing=True)
        
        latent_channels = getattr(model_ddp.module, 'channels', 4)
        shape = (latent_channels, H // 8, W // 8)

        if args.guess_mode:
            model_ddp.module.control_scales = [args.strength * (0.825 ** float(12 - i)) for i in range(13)]
        else:
            model_ddp.module.control_scales = [args.strength] * 13

        print(f"[Rank {rank}/GPU {actual_gpu_id}] Starting DDIM sampling for {total_generations_this_batch} images in batch...")
        with autocast(enabled=args.use_fp16):
            samples, _ = ddim_sampler.sample(
                args.ddim_steps,
                total_generations_this_batch,
                shape,
                cond,
                verbose=False, 
                eta=args.eta,
                unconditional_guidance_scale=args.scale,
                unconditional_conditioning=un_cond
            )
        del cond, un_cond, control_this_gpu
        if args.low_vram: torch.cuda.empty_cache()

        if args.low_vram:
            model_ddp.module.low_vram_shift(device='cpu', is_diffusing=True)
            torch.cuda.empty_cache()

        if args.low_vram:
            model_ddp.module.low_vram_shift(device=actual_gpu_id, is_diffusing=False)

        with autocast(enabled=args.use_fp16):
            x_samples = model_ddp.module.decode_first_stage(samples)
        del samples
        if args.low_vram: torch.cuda.empty_cache()

        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # --- Save batch output ---
        for i in range(num_unique_images_in_batch):
            base_name = batch_base_names[i]
            ext = batch_exts[i]
            x_start, y_start, x_end, y_end = batch_crop_boxes[i]
            
            for sample_idx in range(args.num_samples):
                result_idx = i * args.num_samples + sample_idx
                generated_image_on_canvas_np = x_samples[result_idx]
                cropped_output_content_np = generated_image_on_canvas_np[y_start:y_end, x_start:x_end, :]
                output_image_pil = Image.fromarray(cropped_output_content_np)
                
                if args.num_samples > 1:
                    output_filename = f"generated_from_{base_name}_sample_{sample_idx}{ext}"
                else:
                    output_filename = f"generated_from_{base_name}{ext}"

                current_output_path = os.path.join(args.output_dir, output_filename)
                output_image_pil.save(current_output_path)
                print(f"[Rank {rank}/GPU {actual_gpu_id}] Saved cropped image: {current_output_path} (size: {output_image_pil.size})")

        del x_samples
        if rank == 0 and (args.low_vram or args.enable_sliced_attention):
             torch.cuda.empty_cache()

        batch_process_end_time = time.time()
        batch_duration = batch_process_end_time - batch_process_start_time
        total_image_processing_time += batch_duration
        print(f"--- [Rank {rank}] Finished batch {batch_idx + 1}. Duration: {batch_duration:.2f} seconds. ---\n")


    # 4. Worker Completion Summary
    worker_end_time = time.time()
    worker_total_duration = worker_end_time - worker_start_time
    images_processed_count = len(files_for_this_gpu)
    
    # Final cache clear
    torch.cuda.empty_cache()

    print(f"======== [Rank {rank}/GPU {actual_gpu_id}] DDP worker finished ========")
    print(f"Total duration: {worker_total_duration:.2f} seconds.")
    print(f"  - Model loading took: {model_load_duration:.2f} seconds.")
    if images_processed_count > 0:
        avg_time_per_image = total_image_processing_time / images_processed_count
        print(f"  - Processed {images_processed_count} images, avg time per image: {avg_time_per_image:.2f} seconds.")
    
    print(f"[Rank {rank}/GPU {actual_gpu_id}] Cleaning up DDP environment.")
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description="DDP Inference with ControlNet and Segmentation Maps")
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='./configs/my_scribble_config.yaml',
        help='Path to the training configuration YAML file'
    )
    parser.add_argument(
        '--decoder_model_path',
        type=str,
        required=True,
        help='Path to the trained decoder model checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to an input image or a folder containing multiple input images.'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None, # Now optional
        help='Text prompt for image generation. Can be omitted if --prompt_json_path is provided.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./generated_images',
        help='Folder path to save the generated images.'
    )
    parser.add_argument(
        '--image_resolution', 
        type=int, 
        default=512, 
        help='Target canvas resolution. The long edge of the input image is scaled to this value, and the content is centered on a square canvas of this size for generation. The final output is cropped from this canvas to match the original aspect ratio.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Number of different images to process at once per GPU. >1 can improve GPU utilization but increases VRAM usage.'
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=1, 
        help='Number of samples to generate for each input image'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, # Default to a fixed seed for DDP, can be -1 for random per run
        help='Random seed, -1 for random (each DDP process will derive a seed from this and its rank)'
    )
    parser.add_argument(
        '--ddim_steps', 
        type=int, 
        default=50,
        help='Number of DDIM sampling steps'
    )
    parser.add_argument(
        '--scale', 
        type=float, 
        default=9.0, 
        help='Guidance Scale'
    )
    parser.add_argument(
        '--strength', 
        type=float, 
        default=1.0, 
        help='Control Strength'
    )
    parser.add_argument(
        '--eta', 
        type=float, 
        default=0.0, 
        help='DDIM eta (0.0 corresponds to DDIM, 1.0 to DDPM)'
    )
    parser.add_argument(
        '--a_prompt', 
        type=str, 
        default='best quality, extremely detailed', 
        help='Added Prompt'
    )
    parser.add_argument(
        '--n_prompt', 
        type=str, 
        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
        help='Negative Prompt'
    )
    parser.add_argument(
        '--guess_mode',
        action='store_true',
        help='Enable Guess Mode'
    )
    parser.add_argument(
        '--prompt_json_path',
        type=str,
        default=None,
        help='Path to a JSON file containing prompts for each image. If provided, it will override the --prompt argument for corresponding image files.'
    )
    # DDP specific arguments
    parser.add_argument(
        '--gpu_ids', 
        type=str, 
        default=None, 
        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). Defaults to all available GPUs.'
    )
    parser.add_argument(
        '--ddp_port',
        type=str,
        default="12355",
        help='Port for the DDP master process'
    )
    # Memory optimization arguments
    parser.add_argument(
        '--low_vram',
        action='store_true',
        help='Enable the low_vram_shift mechanism to reduce VRAM usage.'
    )
    parser.add_argument(
        '--enable_sliced_attention',
        action='store_true',
        help='Enable sliced attention to reduce VRAM usage.'
    )
    parser.add_argument(
        '--use_fp16',
        action='store_true',
        help='Enable FP16 (mixed-precision) inference for acceleration and reduced VRAM (may affect precision).'
    )
    parser.add_argument(
        '--use_student_model',
        action='store_true',
        help='Enable student ControlNet model for inference. If not specified, the teacher model will be used.'
    )

    args = parser.parse_args()

    # --- Argument validation and Prompt data loading ---
    if not args.prompt and not args.prompt_json_path:
        print("Error: You must provide either --prompt or --prompt_json_path.")
        sys.exit(1)

    prompt_data = None
    if args.prompt_json_path:
        print(f"Loading prompts from JSON file: {args.prompt_json_path}")
        try:
            # Try 'utf-8-sig' (handles BOM), 'utf-8', 'gbk'
            loaded = False
            for encoding in ['utf-8-sig', 'utf-8', 'gbk']:
                try:
                    with open(args.prompt_json_path, 'r', encoding=encoding) as f:
                        prompt_data = json.load(f)
                    print(f"Successfully loaded prompts from {args.prompt_json_path} with {encoding} encoding.")
                    loaded = True
                    break
                except UnicodeDecodeError:
                    print(f"Failed to decode with {encoding}, trying next encoding...")
                    continue
            
            if not loaded:
                print(f"Error: Could not decode JSON file after trying multiple encodings. Please check the file encoding.")
                sys.exit(1)

        except FileNotFoundError:
            print(f"Error: Prompt JSON file not found: {args.prompt_json_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse prompt JSON file: {args.prompt_json_path}. Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unknown error occurred while reading the prompt JSON file: {e}")
            sys.exit(1)
    
    # Attach prompt data to the args object for passing to the ddp worker
    args.prompt_data = prompt_data
    # ---

    # 1. Collect input files
    input_files = []
    if os.path.isfile(args.input_path):
        input_files.append(args.input_path)
    elif os.path.isdir(args.input_path):
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        for fname in sorted(os.listdir(args.input_path)): # Sort for deterministic processing order
            if os.path.splitext(fname)[1].lower() in image_extensions:
                input_files.append(os.path.join(args.input_path, fname))
    else:
        print(f"Error: Input path '{args.input_path}' is not a valid file or directory.")
        sys.exit(1)

    if not input_files:
        print(f"Error: No image files found in '{args.input_path}'.")
        sys.exit(1)
    
    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # Add the file list to args for passing to DDP workers
    args.input_files = input_files

    if args.enable_sliced_attention:
        # enable_sliced_attention seems to globally patch attention layers,
        # so calling it once in the main process should be sufficient.
        print("Main process: Enabling Sliced Attention.")
        enable_sliced_attention()

    if args.gpu_ids:
        try:
            actual_gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids 
            world_size = torch.cuda.device_count() 
            print(f"Using specified GPUs: {actual_gpu_ids} (CUDA_VISIBLE_DEVICES='{args.gpu_ids}') -> Effective world size: {world_size}")
            if world_size != len(actual_gpu_ids):
                 print(f"Warning: CUDA_VISIBLE_DEVICES ('{args.gpu_ids}') resulted in {world_size} visible GPUs, but {len(actual_gpu_ids)} were specified.")
                 actual_gpu_ids = list(range(world_size))
        except ValueError:
            print("Error: Invalid gpu_ids format. Please use a comma-separated list of integers, e.g., '0,1'.")
            sys.exit(1)
    else:
        world_size = torch.cuda.device_count()
        actual_gpu_ids = list(range(world_size))
        # Construct a CUDA_VISIBLE_DEVICES string for all available GPUs to pass to workers for consistency
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, actual_gpu_ids))
        print(f"Using all available GPUs: {actual_gpu_ids} (CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}', world_size={world_size})")

    if world_size == 0:
        print("Error: No GPUs available.")
        sys.exit(1)
    
    total_jobs = len(input_files)
    if total_jobs < world_size and total_jobs > 0:
        print(f"Warning: Number of images ({total_jobs}) is less than the number of GPUs ({world_size}). Some GPUs will be idle.")

    print(f"Found {total_jobs} total images, will generate {args.num_samples} samples for each.")
    print(f"Using {world_size} GPUs for distributed processing.")
    
    mp.spawn(ddp_worker,
             args=(world_size, args, actual_gpu_ids),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    # Example Usage:
    # python process_decoding.py \
    #   --decoder_model_path "CKPT_PATH" \
    #   --input_path "PATH_TO_IMAGE_OR_FOLDER" \
    #   --prompt "A beautiful landscape" \
    #   --output_dir "./generated_images" \
    #   --num_samples 1 --low_vram --enable_sliced_attention --gpu_ids "0" \
    #   --use_fp16
    main() 