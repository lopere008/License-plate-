import unittest

import numpy as np

from plate_ocr.config import CHAR_SIZE
from plate_ocr.preprocessing import as_gray, binarize, normalize_char
from plate_ocr.synthetic import render_plate


class PreprocessingTest(unittest.TestCase):
    def test_as_gray_normalizes_uint8(self):
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(as_gray(image).max(), 1.0)

    def test_binarize_finds_dark_characters(self):
        mask = binarize(render_plate("AB1234"), "dark")
        self.assertTrue(mask[30:80, 18:300].any())

    def test_normalize_char_shape(self):
        raw = np.ones((60, 40))
        raw[20:45, 10:30] = 0.0
        normalized = normalize_char(raw)
        self.assertEqual(normalized.shape, CHAR_SIZE)
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
