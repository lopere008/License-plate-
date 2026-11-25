from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC


class CharClassifier:
    def __init__(self, c=10.0, gamma="scale"):
        self.model = SVC(C=c, gamma=gamma, kernel="rbf")

    def fit(self, images, labels):
        self.model.fit(self._features(images), labels)
        return self

    def predict(self, images):
        return list(self.model.predict(self._features(images)))

    def _features(self, images):
        return np.asarray(images).reshape(len(images), -1)

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, destination)

    @classmethod
    def load(cls, path):
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {source}. Lancez d'abord python train.py"
            )
        classifier = cls()
        classifier.model = joblib.load(source)
        return classifier
