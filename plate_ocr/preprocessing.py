import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.transform import resize

from .config import CHAR_SIZE


def remove_small_components(binary, min_pixels):
    labeled, count = ndimage.label(binary)
    sizes = np.bincount(labeled.ravel(), minlength=count + 1)
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labeled]


def as_gray(image):
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    array = array.astype(np.float64)
    if array.size and array.max() > 1.0:
        array /= 255.0
    return np.clip(array, 0.0, 1.0)


def binarize(gray, foreground="dark"):
    threshold = threshold_otsu(gray)
    if foreground == "dark":
        return gray < threshold
    return gray > threshold


def normalize_char(char_image):
    gray = as_gray(char_image)
    binary = remove_small_components(binarize(gray, "dark"), max(4, gray.size // 500))
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    if rows.size and cols.size:
        crop = gray[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    else:
        crop = gray
    height, width = crop.shape
    side = max(height, width)
    padded = np.ones((side, side), dtype=np.float64)
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    padded[y_offset : y_offset + height, x_offset : x_offset + width] = crop
    resized = resize(padded, CHAR_SIZE, anti_aliasing=True, preserve_range=True)
    darkness = 1.0 - resized
    if darkness.sum() > 0:
        center_y, center_x = ndimage.center_of_mass(darkness)
        row_shift = int(round(CHAR_SIZE[0] / 2 - center_y))
        col_shift = int(round(CHAR_SIZE[1] / 2 - center_x))
        resized = np.roll(resized, (row_shift, col_shift), axis=(0, 1))
    return np.clip(resized, 0.0, 1.0)
