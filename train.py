import argparse

from plate_ocr.classifier import CharClassifier
from plate_ocr.config import (
    DEFAULT_MODEL_PATH,
    PLATE_COUNT,
    SEED,
    TRAIN_SAMPLES_PER_CLASS,
)
from plate_ocr.evaluation import evaluate_chars, evaluate_plates
from plate_ocr.synthetic import generate_char_splits


def main():
    parser = argparse.ArgumentParser(description="Entraîne le classifieur de caractères")
    parser.add_argument("--samples-per-class", type=int, default=TRAIN_SAMPLES_PER_CLASS)
    parser.add_argument("--plates", type=int, default=PLATE_COUNT)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    (train_images, train_labels), (test_images, test_labels) = generate_char_splits(
        args.samples_per_class, SEED
    )
    classifier = CharClassifier().fit(train_images, train_labels)
    classifier.save(args.model)

    char_accuracy = evaluate_chars(classifier, test_images, test_labels)
    plate_char_accuracy, plate_accuracy = evaluate_plates(classifier, args.plates, SEED + 10)
    print(f"accuracy caractères: {char_accuracy:.3f}")
    print(f"accuracy caractères sur plaques: {plate_char_accuracy:.3f}")
    print(f"accuracy plaques complètes: {plate_accuracy:.3f}")
    print(f"modèle: {args.model}")


if __name__ == "__main__":
    main()
