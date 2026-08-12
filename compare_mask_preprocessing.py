#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lpips_utils import IMAGE_EXTENSIONS
from segmentation_preprocessing import resize_semantic_mask


def legacy_letterbox(mask, resolution):
    height, width = mask.shape[:2]
    scale = resolution / max(width, height)
    content_w = max(1, int(round(width * scale)))
    content_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(mask, (content_w, content_h), interpolation=interpolation)
    canvas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    x = (resolution - content_w) // 2
    y = (resolution - content_h) // 2
    canvas[y:y + content_h, x:x + content_w] = resized
    return canvas


def colors(image):
    return {tuple(color) for color in image.reshape(-1, image.shape[-1])}


def collect_images(input_path):
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input does not exist: {input_path}")
    return [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Compare legacy and corrected semantic-mask preprocessing."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("mask_resize_comparison.csv"))
    args = parser.parse_args()

    image_paths = collect_images(args.input_path)
    if not image_paths:
        parser.error(f"No images found in {args.input_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in image_paths:
        original = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        old = legacy_letterbox(original, args.resolution)
        new = resize_semantic_mask(original, args.resolution)
        original_colors = colors(original)
        old_colors = colors(old)
        new_colors = colors(new)
        rows.append(
            {
                "image": str(path),
                "original_colors": len(original_colors),
                "legacy_colors": len(old_colors),
                "corrected_colors": len(new_colors),
                "legacy_new_colors": len(old_colors - original_colors),
                "corrected_new_colors": len(new_colors - original_colors),
                "changed_pixel_ratio": float(np.mean(np.any(old != new, axis=2))),
            }
        )

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "changed_pixel_ratio": f"{row['changed_pixel_ratio']:.8f}"})
    print(f"Compared {len(rows)} masks; report saved to {args.output}")


if __name__ == "__main__":
    main()
