import unittest

import numpy as np

from plate_ocr.detection import PlateNotFoundError, find_plate
from plate_ocr.synthetic import random_plate, render_car_scene


def intersection_over_union(first, second):
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union


class DetectionTest(unittest.TestCase):
    def test_detects_plate_in_synthetic_scene(self):
        rng = np.random.default_rng(7)
        scores = []
        for _ in range(8):
            image, expected = render_car_scene(random_plate(rng), rng)
            detected = find_plate(image)
            scores.append(intersection_over_union(detected, expected))
        self.assertGreater(np.mean(scores), 0.5)

    def test_raises_on_uniform_image(self):
        with self.assertRaises(PlateNotFoundError):
            find_plate(np.ones((240, 320)))
