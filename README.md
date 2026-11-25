# License Plate Recognition System

Prototype MVP de reconnaissance de plaques d'immatriculation béninoises en Python.

## Format ciblé

Plaque claire, caractères noirs, format `A1234` (ancien) ou `AB1234` (récent) : une ou deux lettres suivies de quatre chiffres, avec un bandeau « RB » à droite. Les images d'entraînement sont générées synthétiquement avec PIL.

## Installation

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

## Utilisation

```bash
venv/bin/python train.py
venv/bin/python predict.py car1.jpg
```

`train.py` entraîne le SVM, écrit `models/svm_plate.joblib` et affiche l'accuracy sur caractères et sur plaques (environ 0,93 sur le jeu synthétique de référence). Le modèle n'est pas versionné. `car1.jpg` n'est pas fourni par le dépôt.

## Structure

- `plate_ocr/synthetic.py` : génération de plaques, de caractères et de scènes de test
- `plate_ocr/detection.py` : détection de la région plaque
- `plate_ocr/segmentation.py` : découpage en caractères
- `plate_ocr/classifier.py` : SVM et persistance joblib
- `plate_ocr/pipeline.py` : lecture complète d'une image
- `tests/` : tests `unittest`

## Tests

```bash
venv/bin/python -m unittest discover -s tests -v
```

Le test `examples/manifest.csv` est ignoré tant que le fichier n'existe pas. Le format attendu est `filename,plate` avec une ligne par image.

## Limites connues

- Entraînement entièrement synthétique : précision incertaine sur photos réelles (nuit, flou, angle, reflets).
- Le bandeau de droite est exclu par un recadrage propre au format béninois.
- Les paires O/0 et I/1 peuvent être confondues.
- Aucune API ni transmission réseau dans ce dépôt.

## Suite prévue

Pilote sur une ou deux caméras, collecte d'images réelles annotées dans `examples/`, calibrage des masques par site, puis passage progressif à une vingtaine de sites avec envoi de la lecture et d'une image nette vers un serveur de visualisation.
