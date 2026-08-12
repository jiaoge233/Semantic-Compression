import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from lpips_utils import calculate_lpips_distance, pair_images


class RecordingLoss:
    def __init__(self):
        self.normalize = None
        self.ranges = None

    def __call__(self, reference, reconstruction, normalize=False):
        self.normalize = normalize
        self.ranges = (
            (reference.min().item(), reference.max().item()),
            (reconstruction.min().item(), reconstruction.max().item()),
        )
        return torch.tensor([0.25])


class LpipsUtilsTest(unittest.TestCase):
    def test_standard_normalization_is_explicit(self):
        loss = RecordingLoss()
        black = Image.new("RGB", (2, 2), (0, 0, 0))
        white = Image.new("RGB", (2, 2), (255, 255, 255))

        value = calculate_lpips_distance(loss, black, white, torch.device("cpu"))

        self.assertEqual(value, 0.25)
        self.assertTrue(loss.normalize)
        self.assertEqual(loss.ranges, ((0.0, 0.0), (1.0, 1.0)))

    def test_pairing_rejects_duplicate_numeric_ids(self):
        with tempfile.TemporaryDirectory() as root:
            reference = Path(root) / "reference"
            reconstruction = Path(root) / "reconstruction"
            reference.mkdir()
            reconstruction.mkdir()
            Image.new("RGB", (1, 1)).save(reference / "1_image.png")
            Image.new("RGB", (1, 1)).save(reference / "copy_1.png")
            Image.new("RGB", (1, 1)).save(reconstruction / "1_result.png")

            with self.assertRaisesRegex(ValueError, "Duplicate numeric image ID"):
                pair_images(reference, reconstruction)

    def test_pairing_rejects_empty_intersection(self):
        with tempfile.TemporaryDirectory() as root:
            reference = Path(root) / "reference"
            reconstruction = Path(root) / "reconstruction"
            reference.mkdir()
            reconstruction.mkdir()
            Image.new("RGB", (1, 1)).save(reference / "1_image.png")
            Image.new("RGB", (1, 1)).save(reconstruction / "2_result.png")

            with self.assertRaisesRegex(ValueError, "No image pairs"):
                pair_images(reference, reconstruction)


if __name__ == "__main__":
    unittest.main()
