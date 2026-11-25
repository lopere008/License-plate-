import argparse

from skimage.io import imread

from plate_ocr.classifier import CharClassifier
from plate_ocr.config import DEFAULT_MODEL_PATH
from plate_ocr.pipeline import recognize


def main():
    parser = argparse.ArgumentParser(description="Lit la plaque d'une image")
    parser.add_argument("image")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    classifier = CharClassifier.load(args.model)
    result = recognize(imread(args.image), classifier)
    print(result.text)
    print("format valide" if result.valid else "format invalide")
    print(f"bbox: {result.bbox}")


if __name__ == "__main__":
    main()
