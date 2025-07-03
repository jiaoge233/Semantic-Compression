import argparse
import os
import torch
import lpips
import csv
import re
import shutil
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.nn.parallel import DataParallel
from torch_fidelity import calculate_metrics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import imquality.brisque as brisque
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def preprocess_images(source_dir, dest_dir, size):
    """
    Resizes all images in source_dir to a standard size and saves them in dest_dir.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    resize_transform = transforms.Resize((size, size))

    print(f"Preprocessing images from {source_dir} to {dest_dir} with size {size}x{size}...")
    for f in os.listdir(source_dir):
        source_path = os.path.join(source_dir, f)
        if os.path.isfile(source_path):
            try:
                img = Image.open(source_path).convert("RGB")
                img_resized = resize_transform(img)
                # Preserve original filename by appending .png, and save as PNG for FID compatibility
                dest_path = os.path.join(dest_dir, f + ".png")
                img_resized.save(dest_path, format='PNG')
            except Exception as e:
                print(f"Could not process and save image {source_path}: {e}")

def get_image_map(directory):
    """
    Creates a map from a numeric prefix of a filename to the full path of the file.
    """
    image_map = {}
    for f in sorted(os.listdir(directory)):
        if os.path.isfile(os.path.join(directory, f)):
            match = re.search(r'\d+', f)
            if match:
                num_prefix = int(match.group(0))
                image_map[num_prefix] = os.path.join(directory, f)
    return image_map

def main(args):
    # Create temporary directories for resized images
    temp_dir1 = "temp_resized_dir1_for_metrics"
    temp_dir2 = "temp_resized_dir2_for_metrics"

    try:
        # Preprocess both directories to a standard size
        preprocess_images(args.dir1, temp_dir1, args.image_size)
        preprocess_images(args.dir2, temp_dir2, args.image_size)

        # Setup GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Cosine Similarity Setup ---
        print("Setting up model for Cosine Similarity calculation...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for Cosine Similarity!")
            resnet = DataParallel(resnet)
        resnet.to(device)
        resnet.eval()

        # Define the image transformation for ResNet
        preprocess_for_resnet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- LPIPS Calculation ---
        print("Calculating LPIPS...")
        lpips_per_pair_file = 'lpips_per_pair.csv'
        summary_file = 'summary_metrics.csv'

        loss_fn_alex = lpips.LPIPS(net='alex')
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for LPIPS!")
            loss_fn_alex = DataParallel(loss_fn_alex)
        loss_fn_alex.to(device)

        # Pair images based on numerical prefixes in their filenames from temp directories
        print("Pairing images based on numerical prefixes...")
        image_map1 = get_image_map(temp_dir1)
        image_map2 = get_image_map(temp_dir2)

        image_pairs = []
        for key in sorted(image_map1.keys()):
            if key in image_map2:
                image_pairs.append((image_map1[key], image_map2[key]))
        
        print(f"Found {len(image_pairs)} image pairs with matching numerical prefixes.")

        total_lpips, total_ssim, total_psnr, total_cosine = 0.0, 0.0, 0.0, 0.0
        num_pairs = 0

        with open(lpips_per_pair_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image1', 'image2', 'lpips', 'ssim', 'psnr', 'cosine_similarity'])

            total_pairs = len(image_pairs)
            print(f"Starting LPIPS/SSIM/PSNR/Cosine Similarity calculation for {total_pairs} pairs...")
            for i, (img_path1, img_path2) in enumerate(image_pairs):
                try:
                    # Images are already resized, just open and convert
                    img1 = Image.open(img_path1).convert("RGB")
                    img2 = Image.open(img_path2).convert("RGB")
                    
                    img1_np = np.array(img1)
                    img2_np = np.array(img2)
                    
                    # LPIPS
                    transform_tensor = transforms.ToTensor()
                    img1_tensor = transform_tensor(img1).to(device).unsqueeze(0)
                    img2_tensor = transform_tensor(img2).to(device).unsqueeze(0)
                    with torch.no_grad():
                        lpips_dist = loss_fn_alex(img1_tensor, img2_tensor).item()
                    
                    # SSIM
                    ssim_value = structural_similarity(img1_np, img2_np, channel_axis=-1, data_range=255)

                    # PSNR
                    psnr_value = peak_signal_noise_ratio(img1_np, img2_np, data_range=255)

                    # Cosine Similarity
                    cosine_sim = 0.0
                    with torch.no_grad():
                        img1_res_tensor = preprocess_for_resnet(img1).unsqueeze(0).to(device)
                        img2_res_tensor = preprocess_for_resnet(img2).unsqueeze(0).to(device)
                        features1 = resnet(img1_res_tensor).squeeze()
                        features2 = resnet(img2_res_tensor).squeeze()
                        cosine_sim = F.cosine_similarity(features1.unsqueeze(0), features2.unsqueeze(0)).item()

                    # Report original paths in the CSV for clarity
                    # The temporary files have .png appended, so we remove it to get the original name.
                    original_basename1 = os.path.basename(img_path1)
                    if original_basename1.lower().endswith('.png'):
                        original_basename1 = original_basename1[:-4]
                    
                    original_basename2 = os.path.basename(img_path2)
                    if original_basename2.lower().endswith('.png'):
                        original_basename2 = original_basename2[:-4]

                    original_path1 = os.path.join(args.dir1, original_basename1)
                    original_path2 = os.path.join(args.dir2, original_basename2)
                    writer.writerow([original_path1, original_path2, f'{lpips_dist:.4f}', f'{ssim_value:.4f}', f'{psnr_value:.4f}', f'{cosine_sim:.4f}'])
                    
                    total_lpips += lpips_dist
                    total_ssim += ssim_value
                    total_psnr += psnr_value
                    total_cosine += cosine_sim
                    num_pairs += 1
                    
                    if (i + 1) % 10 == 0 or (i + 1) == total_pairs:
                        print(f"  Processed {i + 1} / {total_pairs} pairs...")
                except Exception as e:
                    print(f"Skipping pair due to error: {img_path1}, {img_path2} - {e}")

        avg_lpips, avg_ssim, avg_psnr, avg_cosine = 0, 0, 0, 0
        if num_pairs > 0:
            avg_lpips = total_lpips / num_pairs
            avg_ssim = total_ssim / num_pairs
            avg_psnr = total_psnr / num_pairs
            avg_cosine = total_cosine / num_pairs
            print(f"Average LPIPS: {avg_lpips:.4f}")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Average PSNR: {avg_psnr:.4f}")
            print(f"Average Cosine Similarity: {avg_cosine:.4f}")
            print(f"Per-pair results saved to {lpips_per_pair_file}")
        else:
            print("No image pairs found to calculate LPIPS/SSIM/PSNR.")
            
        # --- BRISQUE Calculation for dir1 ---
        print("\nCalculating BRISQUE for dir1...")
        brisque_file = 'brisque_scores_dir1.csv'
        brisque_scores = []
        avg_brisque = 0

        with open(brisque_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image', 'brisque'])

            image_paths1 = list(image_map1.values())
            total_imgs1 = len(image_paths1)
            if total_imgs1 > 0:
                print(f"Starting BRISQUE calculation for {total_imgs1} images in dir1...")

                for i, img_path in enumerate(image_paths1):
                    try:
                        # Use the already resized images from the temp directory
                        img_np = np.array(Image.open(img_path).convert("RGB"))
                        brisque_score = brisque.score(img_np)
                        
                        original_basename = os.path.basename(img_path)
                        if original_basename.lower().endswith('.png'):
                            original_basename = original_basename[:-4]

                        original_path = os.path.join(args.dir1, original_basename)
                        writer.writerow([original_path, f'{brisque_score:.4f}'])
                        brisque_scores.append(brisque_score)
                        if (i + 1) % 10 == 0 or (i + 1) == total_imgs1:
                            print(f"  Processed {i + 1} / {total_imgs1} images for BRISQUE...")
                    except Exception as e:
                        print(f"Skipping BRISQUE calculation for {img_path} due to error: {e}")

        if brisque_scores:
            avg_brisque = np.mean(brisque_scores)
            print(f"Average BRISQUE (dir1): {avg_brisque:.4f}")
            print(f"Per-image BRISQUE results for dir1 saved to {brisque_file}")
        else:
            print("No images found in dir1 to calculate BRISQUE.")

        # --- FID Calculation ---
        print("\nCalculating FID...")
        fid_score = "N/A"
        try:
            metrics_dict = calculate_metrics(
                input1=temp_dir1, # Use resized images
                input2=temp_dir2, # Use resized images
                cuda=torch.cuda.is_available(),
                isc=False,
                fid=True,
                kid=False,
                verbose=True,
                samples_find_deep=True
            )
            fid_score = metrics_dict['frechet_inception_distance']
            print(f"FID: {fid_score:.4f}")
        except Exception as e:
            print(f"Could not calculate FID: {e}")

        # --- Save Summary Metrics ---
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'value'])
            writer.writerow(['average_lpips', f'{avg_lpips:.4f}'])
            writer.writerow(['average_ssim', f'{avg_ssim:.4f}'])
            writer.writerow(['average_psnr', f'{avg_psnr:.4f}'])
            writer.writerow(['average_cosine_similarity', f'{avg_cosine:.4f}'])
            writer.writerow(['average_brisque_dir1', f'{avg_brisque:.4f}'])
            if isinstance(fid_score, float):
                writer.writerow(['overall_fid', f'{fid_score:.4f}'])
            else:
                writer.writerow(['overall_fid', fid_score])
        
        print(f"Summary metrics saved to {summary_file}")

    finally:
        # Cleanup temporary directories
        print("Cleaning up temporary directories...")
        if os.path.exists(temp_dir1):
            shutil.rmtree(temp_dir1)
        if os.path.exists(temp_dir2):
            shutil.rmtree(temp_dir2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LPIPS and FID between two image directories.")
    parser.add_argument('dir1', type=str, help='Path to the first image directory.')
    parser.add_argument('dir2', type=str, help='Path to the second image directory.')
    parser.add_argument('--image-size', type=int, default=256, help='The size to resize all images to for calculation.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU(s) to use, e.g., "0" or "0,1"')
    args = parser.parse_args()
    main(args) 