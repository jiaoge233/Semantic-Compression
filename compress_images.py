import os
import io
import argparse
from PIL import Image

def compress_to_target_size(image_path, output_path, target_kb, format='jpeg', max_iterations=15, tolerance=0.5):
    """
    Compresses an image to a target size in kilobytes.

    :param image_path: Path to the input image.
    :param output_path: Path to save the compressed image.
    :param target_kb: The target file size in kilobytes.
    :param format: The output format ('jpeg', 'webp', or 'png').
    :param max_iterations: Maximum number of iterations for quality/color search.
    :param tolerance: The tolerance in KB to accept a file size.
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"无法打开图片 {image_path}: {e}")
        return

    target_bytes = target_kb * 1024
    format_lower = format.lower()

    # For JPEG and WebP, we search for the best quality setting.
    if format_lower in ['jpeg', 'webp']:
        # Convert to RGB if necessary, as JPEG does not support alpha channel
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert("RGB")

        low_quality = 1
        high_quality = 100
        best_quality = -1
        best_image_data = None

        for i in range(max_iterations):
            if high_quality < low_quality:
                break

            quality = (low_quality + high_quality) // 2
            if quality == 0:
                break
            
            buffer = io.BytesIO()
            try:
                if format_lower == 'jpeg':
                    img.save(buffer, format='JPEG', quality=quality, optimize=True)
                elif format_lower == 'webp':
                    img.save(buffer, format='WEBP', quality=quality)
            except Exception as e:
                print(f"压缩时发生错误 (quality={quality}): {e}")
                high_quality = quality - 1
                continue

            current_size = buffer.tell()

            if current_size <= target_bytes:
                best_quality = quality
                best_image_data = buffer.getvalue()
                low_quality = quality + 1
            else:
                high_quality = quality - 1

        if best_image_data:
            with open(output_path, 'wb') as f:
                f.write(best_image_data)
            final_size_kb = len(best_image_data) / 1024
            print(f"成功: {os.path.basename(image_path)} -> {os.path.basename(output_path)} | 大小: {final_size_kb:.2f}KB | 质量: {best_quality}")
        else:
            print(f"失败: 无法将 {os.path.basename(image_path)} (as {format}) 压缩到 {target_kb}KB 以下。")

    elif format_lower == 'png':
        # For PNG, we reduce the number of colors (quantization) to reduce size.
        low_colors = 2
        high_colors = 256
        best_colors = -1
        best_image_data = None

        for i in range(max_iterations):
            if high_colors < low_colors:
                break
            
            num_colors = (low_colors + high_colors) // 2
            if num_colors < 2:
                break
            
            buffer = io.BytesIO()
            try:
                # Quantize the image to reduce the number of colors
                quantized_img = img.quantize(colors=num_colors, method=Image.Quantize.LIBIMAGEQUANT, dither=Image.Dither.FLOYDSTEINBERG)
                quantized_img.save(buffer, format='PNG', optimize=True)
            except Exception:
                # Fallback to default quantizer if libimagequant is not available
                try:
                    quantized_img = img.quantize(colors=num_colors, dither=Image.Dither.FLOYDSTEINBERG)
                    quantized_img.save(buffer, format='PNG', optimize=True)
                except Exception as e2:
                    print(f"PNG 压缩失败 (colors={num_colors}): {e2}")
                    high_colors = num_colors - 1
                    continue

            current_size = buffer.tell()

            if current_size <= target_bytes:
                best_colors = num_colors
                best_image_data = buffer.getvalue()
                # Try to improve quality with more colors
                low_colors = num_colors + 1
            else:
                high_colors = num_colors - 1

        if best_image_data:
            with open(output_path, 'wb') as f:
                f.write(best_image_data)
            final_size_kb = len(best_image_data) / 1024
            print(f"成功: {os.path.basename(image_path)} -> {os.path.basename(output_path)} | 大小: {final_size_kb:.2f}KB | 颜色数: {best_colors}")
        else:
            print(f"失败: 无法将 {os.path.basename(image_path)} (as PNG) 压缩到 {target_kb}KB 以下。")
    else:
        print(f"不支持的格式: '{format}'. 请选择 'jpeg', 'webp', 或 'png'.")


def main():
    parser = argparse.ArgumentParser(description="将图片压缩到指定大小，并为每张图片生成 JPEG, WebP, 和 PNG 三种格式。")
    parser.add_argument('--input-dir', required=True, help="包含原始图片的文件夹路径。")
    parser.add_argument('--output-dir', required=True, help="用于保存压缩后图片的文件夹路径。")
    parser.add_argument('--target-size', type=int, required=True, help="目标文件大小（单位：KB）。")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"错误: 输入文件夹 '{args.input_dir}' 不存在。")
        return

    formats_to_generate = ['jpeg', 'webp', 'png']
    for fmt in formats_to_generate:
        os.makedirs(os.path.join(args.output_dir, fmt), exist_ok=True)

    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    for filename in os.listdir(args.input_dir):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            input_path = os.path.join(args.input_dir, filename)
            
            for fmt in formats_to_generate:
                base_filename = os.path.splitext(filename)[0]
                output_filename = f"{base_filename}.{fmt.lower()}"
                output_path = os.path.join(args.output_dir, fmt, output_filename)
                
                print(f"\n--- 正在处理 {filename} -> {fmt.upper()} ---")
                compress_to_target_size(input_path, output_path, args.target_size, fmt)

if __name__ == '__main__':
    main() 