import numpy as np
from skimage import measure, morphology

from .config import PLATE_ASPECT_RATIO
from .preprocessing import as_gray, binarize, remove_small_components


class PlateNotFoundError(Exception):
    pass


def _candidates(gray, foreground, close_width):
    binary = binarize(gray, foreground)
    if close_width:
        binary = morphology.closing(binary, morphology.footprint_rectangle((1, close_width)))
    binary = remove_small_components(binary, max(20, gray.size // 5000))
    height, width = gray.shape
    candidates = []
    for region in measure.regionprops(measure.label(binary)):
        min_row, min_col, max_row, max_col = region.bbox
        box_width = max_col - min_col
        box_height = max_row - min_row
        if box_width < 40 or box_height < 10:
            continue
        ratio = box_width / box_height
        if not 3.0 <= ratio <= 7.0:
            continue
        if box_width > 0.85 * width or box_height > 0.5 * height:
            continue
        penalty = abs(np.log(ratio / PLATE_ASPECT_RATIO))
        candidates.append((region.area / (1.0 + penalty), (min_col, min_row, max_col, max_row)))
    return candidates


def find_plate(image):
    gray = as_gray(image)
    candidates = _candidates(gray, "light", 3) + _candidates(gray, "dark", 25)
    if not candidates:
        raise PlateNotFoundError("Aucune région de plaque detectable dans l'image")
    candidates.sort(reverse=True)
    _, (x0, y0, x1, y1) = candidates[0]
    return (x0, y0, x1, y1)
