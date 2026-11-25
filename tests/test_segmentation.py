import unittest

from plate_ocr.config import CHAR_SIZE
from plate_ocr.segmentation import segment_chars
from plate_ocr.synthetic import generate_plates


class SegmentationTest(unittest.TestCase):
    def test_segments_expected_number_of_characters(self):
        images, labels = generate_plates(10, 5, augment=False)
        exact = 0
        for image, label in zip(images, labels):
            chars = segment_chars(image)
            self.assertIn(len(chars), (5, 6))
            for char in chars:
                self.assertEqual(char.shape, CHAR_SIZE)
            exact += int(len(chars) == len(label))
        self.assertGreaterEqual(exact, 8)
