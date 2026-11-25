from .pipeline import recognize_plate
from .synthetic import generate_plates


def evaluate_chars(classifier, images, labels):
    predictions = classifier.predict(images)
    return sum(predicted == expected for predicted, expected in zip(predictions, labels)) / len(
        labels
    )


def evaluate_plates(classifier, count, seed):
    images, labels = generate_plates(count, seed, augment=True)
    correct_chars = 0
    total_chars = 0
    correct_plates = 0
    for image, label in zip(images, labels):
        text = recognize_plate(image, classifier).text
        correct_chars += sum(expected == predicted for expected, predicted in zip(label, text))
        total_chars += len(label)
        correct_plates += int(text == label)
    return correct_chars / total_chars, correct_plates / len(images)
