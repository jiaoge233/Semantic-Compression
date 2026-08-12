import unittest

import numpy as np

from segmentation_preprocessing import (
    normalize_control_array,
    preprocess_semantic_mask,
    resize_semantic_mask,
    restore_generated_image,
)


def color_set(image):
    return {tuple(color) for color in image.reshape(-1, image.shape[-1])}


class SegmentationPreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.array(
            [
                [[0, 0, 0], [0, 0, 0], [255, 0, 0]],
                [[0, 0, 0], [0, 255, 0], [255, 0, 0]],
            ],
            dtype=np.uint8,
        )

    def test_nearest_resize_does_not_create_colors(self):
        resized = resize_semantic_mask(self.mask, 11)
        self.assertEqual(resized.shape, (11, 11, 3))
        self.assertTrue(color_set(resized).issubset(color_set(self.mask)))

    def test_array_and_tensor_paths_are_identical(self):
        resized, tensor = preprocess_semantic_mask(self.mask, 8)
        training_array = normalize_control_array(
            resize_semantic_mask(self.mask, 8)
        )
        inference_array = tensor.squeeze(0).permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(resized, resize_semantic_mask(self.mask, 8))
        np.testing.assert_array_equal(training_array, inference_array)

    def test_restore_generated_image_uses_original_dimensions(self):
        generated = np.zeros((8, 8, 3), dtype=np.uint8)
        restored = restore_generated_image(generated, (13, 5))
        self.assertEqual(restored.shape, (5, 13, 3))

    def test_invalid_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            resize_semantic_mask(self.mask, 0)


if __name__ == "__main__":
    unittest.main()
