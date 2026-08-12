import cv2
import numpy as np
import torch


def resize_semantic_mask(mask, resolution):
    """Stretch a discrete semantic mask to a square without creating new colors."""
    if resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    mask = np.asarray(mask)
    if mask.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D mask, got shape {mask.shape}")
    return cv2.resize(
        mask,
        (resolution, resolution),
        interpolation=cv2.INTER_NEAREST,
    )


def normalize_control_array(mask):
    """Normalize an HWC semantic control image to a float32 HWC array."""
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError(f"Expected an RGB mask, got shape {mask.shape}")
    return mask.astype(np.float32) / 127.5 - 1.0


def normalize_control_image(mask):
    """Normalize an HWC semantic control image to a BCHW float tensor."""
    normalized = normalize_control_array(mask)
    return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).contiguous()


def preprocess_semantic_mask(mask, resolution):
    resized = resize_semantic_mask(mask, resolution)
    return resized, normalize_control_image(resized)


def restore_generated_image(image, output_size):
    """Resize a generated RGB square back to (width, height)."""
    output_w, output_h = output_size
    if output_w <= 0 or output_h <= 0:
        raise ValueError("output dimensions must be positive")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    if image.shape[1] == output_w and image.shape[0] == output_h:
        return image
    return cv2.resize(
        image,
        (output_w, output_h),
        interpolation=cv2.INTER_LANCZOS4,
    )
