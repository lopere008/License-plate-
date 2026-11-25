from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from scipy import ndimage

from .config import (
    ALPHABET,
    DIGITS,
    FONT_CANDIDATES,
    LEGACY_PROBABILITY,
    LETTERS,
    PLATE_HEIGHT,
    PLATE_WIDTH,
)
from .preprocessing import normalize_char


def load_font(size):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def random_plate(rng):
    letters_count = 1 if rng.random() < LEGACY_PROBABILITY else 2
    letters = "".join(str(rng.choice(list(LETTERS))) for _ in range(letters_count))
    digits = "".join(str(rng.choice(list(DIGITS))) for _ in range(4))
    return letters + digits


def render_plate(plate, augment=False, rng=None):
    image = Image.new("L", (PLATE_WIDTH, PLATE_HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    letters = plate[:-4]
    digits = plate[-4:]
    draw.text((18, 12), letters + " " + digits, font=load_font(72), fill=0)
    draw.rectangle((392, 10, 512, 100), fill=215)
    draw.rectangle((392, 10, 512, 16), fill=90)
    draw.text((452, 42), "RB", font=load_font(34), fill=0)
    if augment:
        if rng is None:
            rng = np.random.default_rng()
        image = image.rotate(float(rng.uniform(-3, 3)), resample=Image.BICUBIC, fillcolor=255)
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.82, 1.18)))
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.12)))
    array = np.asarray(image, dtype=np.float64) / 255.0
    if augment:
        array += rng.normal(0, 0.025, array.shape)
        if rng.random() < 0.25:
            array = ndimage.gaussian_filter(array, sigma=float(rng.uniform(0.4, 0.9)))
    return np.clip(array, 0.0, 1.0)


def generate_plates(count, seed, augment=True):
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for _ in range(count):
        plate = random_plate(rng)
        images.append(render_plate(plate, augment=augment, rng=rng))
        labels.append(plate)
    return images, labels


def render_char(char):
    canvas = Image.new("L", (120, 120), 255)
    draw = ImageDraw.Draw(canvas)
    font = load_font(84)
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (120 - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (120 - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)
    return np.asarray(canvas, dtype=np.float64) / 255.0


def augment_char(char_image, rng):
    rotated = ndimage.rotate(
        char_image, float(rng.uniform(-8, 8)), reshape=False, cval=1.0, order=1
    )
    array = rotated * float(rng.uniform(0.85, 1.1))
    array += rng.normal(0, 0.04, array.shape)
    return np.clip(array, 0.0, 1.0)


def generate_char_dataset(samples_per_class, seed, augment=True):
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for char in ALPHABET:
        clean = normalize_char(render_char(char))
        for _ in range(samples_per_class):
            images.append(augment_char(clean, rng) if augment else clean.copy())
            labels.append(char)
    return images, labels


def generate_char_splits(samples_per_class, seed, test_fraction=0.2, augment=True):
    images, labels = generate_char_dataset(samples_per_class, seed, augment=augment)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(images))
    test_size = int(len(images) * test_fraction)
    test_indices = order[:test_size]
    train_indices = order[test_size:]
    train = ([images[i] for i in train_indices], [labels[i] for i in train_indices])
    test = ([images[i] for i in test_indices], [labels[i] for i in test_indices])
    return train, test


def render_car_scene(plate, rng, size=(1000, 520)):
    canvas = Image.new("L", size, 205)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((50, 200, 950, 440), fill=95)
    draw.polygon([(150, 200), (220, 100), (780, 100), (850, 200)], fill=75)
    draw.rectangle((90, 270, 160, 315), fill=235)
    draw.rectangle((840, 270, 910, 315), fill=235)
    plate_array = render_plate(plate)
    plate_image = Image.fromarray((plate_array * 255).astype(np.uint8), mode="L")
    x = int(rng.integers(100, 300))
    y = int(rng.integers(280, 350))
    canvas.paste(plate_image, (x, y))
    return (
        np.asarray(canvas, dtype=np.float64) / 255.0,
        (x, y, x + PLATE_WIDTH, y + PLATE_HEIGHT),
    )
