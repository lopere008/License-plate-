import csv
import unittest

import numpy as np
from skimage.io import imread

from plate_ocr.classifier import CharClassifier
from plate_ocr.config import EXAMPLES_MANIFEST
from plate_ocr.evaluation import evaluate_plates
from plate_ocr.pipeline import recognize
from plate_ocr.synthetic import generate_char_splits, random_plate, render_car_scene


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (train_images, train_labels), _ = generate_char_splits(12, 101)
        cls.classifier = CharClassifier().fit(train_images, train_labels)

    def test_accuracy_floor(self):
        char_accuracy, plate_accuracy = evaluate_plates(self.classifier, 30, 202)
        self.assertGreaterEqual(char_accuracy, 0.85)
        self.assertGreaterEqual(plate_accuracy, 0.40)

    def test_end_to_end_on_synthetic_scene(self):
        rng = np.random.default_rng(55)
        plate = random_plate(rng)
        image, _ = render_car_scene(plate, rng)
        result = recognize(image, self.classifier)
        self.assertEqual(result.text, plate)
        self.assertEqual(len(result.bbox), 4)

    def test_examples_manifest(self):
        if not EXAMPLES_MANIFEST.exists():
            self.skipTest("examples/manifest.csv absent")
        with EXAMPLES_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            image_path = EXAMPLES_MANIFEST.parent / row["filename"]
            if not image_path.exists():
                continue
            result = recognize(imread(image_path), self.classifier)
            self.assertEqual(result.text, row["plate"])
