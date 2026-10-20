from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def contains_class_images(directory: Path) -> bool:
    """Return whether a directory contains class subfolders with image files."""
    try:
        for class_dir in directory.iterdir():
            if class_dir.is_dir() and any(
                image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
                for image in class_dir.iterdir()
            ):
                return True
    except OSError:
        return False
    return False


print("Downloading dataset from Kaggle...")
downloaded_path = Path(kagglehub.dataset_download("ananyaarya22/isl-data"))
print(f"Kagglehub downloaded data to: {downloaded_path}")

possible_subfolders = (
    Path("."),
    Path("isl-data"),
    Path("ISL"),
    Path("isl-data") / "isl-data",
)
images_path = next(
    (
        candidate
        for subfolder in possible_subfolders
        if (candidate := downloaded_path / subfolder).is_dir()
        and contains_class_images(candidate)
    ),
    None,
)

if images_path is None:
    raise FileNotFoundError(
        "Could not find class folders containing images in the Kaggle download "
        f"at {downloaded_path}. Expected image files inside category folders."
    )

print(f"Resolved source images path: {images_path}")

features = []
labels = []
target_size = (32, 32)
skipped_images = 0

for class_dir in sorted(images_path.iterdir()):
    if not class_dir.is_dir():
        continue

    image_paths = sorted(
        image
        for image in class_dir.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )[:200]

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image = image.convert("L").resize(target_size)
                features.append(np.asarray(image).flatten())
                labels.append(class_dir.name)
        except OSError:
            skipped_images += 1

if not features:
    raise ValueError(f"No readable images found under {images_path}.")
if skipped_images:
    print(f"Skipped {skipped_images} unreadable image(s).")

X = np.asarray(features)
Y = np.asarray(labels)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, Y_train)
print("Model trained successfully")
