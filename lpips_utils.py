import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def build_numeric_image_map(directory):
    """Index images by the first numeric token in their filename."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")

    image_map = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = re.search(r"\d+", path.name)
        if match is None:
            continue
        image_id = int(match.group(0))
        if image_id in image_map:
            raise ValueError(
                f"Duplicate numeric image ID {image_id} in {directory}: "
                f"{image_map[image_id].name} and {path.name}"
            )
        image_map[image_id] = path
    return image_map


def pair_images(reference_dir, reconstruction_dir):
    reference_map = build_numeric_image_map(reference_dir)
    reconstruction_map = build_numeric_image_map(reconstruction_dir)
    common_ids = sorted(reference_map.keys() & reconstruction_map.keys())
    if not common_ids:
        raise ValueError(
            f"No image pairs with matching numeric IDs: "
            f"{reference_dir} vs {reconstruction_dir}"
        )
    return [
        (image_id, reference_map[image_id], reconstruction_map[image_id])
        for image_id in common_ids
    ]


def image_to_lpips_tensor(image, device):
    """Convert a PIL RGB image to a batched float tensor in [0, 1]."""
    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def calculate_lpips_distance(loss_fn, reference_image, reconstruction_image, device):
    """Calculate standard LPIPS, explicitly normalizing [0, 1] to [-1, 1]."""
    reference_tensor = image_to_lpips_tensor(reference_image, device)
    reconstruction_tensor = image_to_lpips_tensor(reconstruction_image, device)
    with torch.no_grad():
        return loss_fn(
            reference_tensor,
            reconstruction_tensor,
            normalize=True,
        ).item()


def load_rgb_image(path, size=None):
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return image
