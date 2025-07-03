import argparse
from model.encoder import ImageEncoder

def main():
    parser = argparse.ArgumentParser(description="Image Encoder: Process a folder of images for segmentation and annotation.")
    
    # 定义命令行参数
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Path to the input directory with original images.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to the output directory to save results (segmented images and annotations.json).")
    parser.add_argument("--annotator_model_path", type=str, default="liuhaotian/llava-v1.5-13b", 
                        help="Path to the LLaVA model for annotation. Can be a local path or a HuggingFace model name.")
    parser.add_argument("--segmentation_ckpt", type=str, required=True, 
                        help="Path to the segmentation model checkpoint (.pth file).")
    parser.add_argument("--segmentation_palette", type=str, default=None, 
                        help="(Optional) Path to a palette image (e.g., from PASCAL VOC or Cityscapes) for colorizing segmentation maps.")
    
    args = parser.parse_args()

    print("Initializing Image Encoder...")
    encoder = ImageEncoder(
        annotator_model_path=args.annotator_model_path,
        segmentation_ckpt_path=args.segmentation_ckpt,
        segmentation_palette_path=args.segmentation_palette
    )
    
    print(f"Starting processing for folder: {args.input_dir}")
    encoder.process_folder(args.input_dir, args.output_dir)
    print("Processing finished successfully.")

if __name__ == "__main__":
    main() 



