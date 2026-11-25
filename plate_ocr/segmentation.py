from skimage import measure

from .preprocessing import as_gray, binarize, normalize_char, remove_small_components


class SegmentationError(Exception):
    pass


def _projection_boxes(binary):
    active = binary.sum(axis=0) > 0
    runs = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return [(left, 0, right, binary.shape[0]) for left, right in runs]


def segment_chars(plate_roi):
    gray = as_gray(plate_roi)
    height, width = gray.shape
    left_region = gray[:, : int(width * 0.74)]
    dark = binarize(left_region, "dark")
    dark = remove_small_components(dark, max(10, dark.size // 2000))
    regions = measure.regionprops(measure.label(dark, connectivity=2))
    boxes = []
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        box_height = max_row - min_row
        box_width = max_col - min_col
        if box_height < 0.35 * height or box_width < 3 or region.area < 20:
            continue
        if box_width > 0.35 * left_region.shape[1]:
            continue
        boxes.append((min_col, min_row, max_col, max_row))
    boxes.sort()
    merged = []
    for box in boxes:
        if merged and box[0] <= merged[-1][2] - 1 and box[2] >= merged[-1][0]:
            previous = merged[-1]
            merged[-1] = (
                min(previous[0], box[0]),
                min(previous[1], box[1]),
                max(previous[2], box[2]),
                max(previous[3], box[3]),
            )
        else:
            merged.append(box)
    if len(merged) < 5:
        merged = _projection_boxes(dark)
    if len(merged) < 5:
        raise SegmentationError(f"Seulement {len(merged)} caractères détectés")
    chars = []
    for x0, y0, x1, y1 in merged[:6]:
        pad = 3
        crop = gray[
            max(0, y0 - pad) : min(height, y1 + pad),
            max(0, x0 - pad) : min(left_region.shape[1], x1 + pad),
        ]
        chars.append(normalize_char(crop))
    return chars
