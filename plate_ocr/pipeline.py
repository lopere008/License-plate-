import re
from dataclasses import dataclass

from .classifier import CharClassifier
from .config import DEFAULT_MODEL_PATH
from .detection import find_plate
from .preprocessing import as_gray
from .segmentation import segment_chars

PLATE_PATTERN = re.compile(r"^[A-Z]{1,2}\d{4}$")


class RecognitionError(Exception):
    pass


@dataclass
class PlateResult:
    text: str
    bbox: tuple
    crop: object
    valid: bool


def recognize_plate(plate_image, classifier):
    chars = segment_chars(plate_image)
    if len(chars) not in (5, 6):
        raise RecognitionError(f"{len(chars)} caractères détectés")
    text = "".join(classifier.predict(chars))
    return PlateResult(
        text=text,
        bbox=None,
        crop=as_gray(plate_image),
        valid=bool(PLATE_PATTERN.match(text)),
    )


def recognize(image, classifier=None, model_path=DEFAULT_MODEL_PATH):
    if classifier is None:
        classifier = CharClassifier.load(model_path)
    x0, y0, x1, y1 = find_plate(image)
    crop = as_gray(image)[y0:y1, x0:x1]
    result = recognize_plate(crop, classifier)
    result.bbox = (x0, y0, x1, y1)
    return result
