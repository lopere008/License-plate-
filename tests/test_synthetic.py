import unittest

import numpy as np

from plate_ocr.config import ALPHABET, PLATE_HEIGHT, PLATE_WIDTH
from plate_ocr.synthetic import generate_plates, random_plate


class SyntheticTest(unittest.TestCase):
    def test_random_plate_format(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            self.assertRegex(random_plate(rng), r"^[A-Z]{1,2}\d{4}$")

    def test_generate_plates_is_reproducible(self):
        first_images, first_labels = generate_plates(4, 3, augment=True)
        second_images, second_labels = generate_plates(4, 3, augment=True)
        self.assertEqual(first_labels, second_labels)
        np.testing.assert_allclose(np.stack(first_images), np.stack(second_images))

    def test_generated_shapes_and_labels(self):
        images, labels = generate_plates(4, 2, augment=False)
        self.assertEqual(len(images), 4)
        self.assertEqual(images[0].shape, (PLATE_HEIGHT, PLATE_WIDTH))
        self.assertTrue(all(char in ALPHABET for label in labels for char in label))
