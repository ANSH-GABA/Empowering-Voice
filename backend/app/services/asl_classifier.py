"""Runs the pre-trained ASL ResNet18 classifier (99.72% validation accuracy).

Much faster than ViT-base (~15ms on CPU vs ~250ms).
"""
import io
import os
from functools import lru_cache
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import models, transforms

MODEL_SNAPSHOT_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--huzaifanasirrr--realtime-sign-language-translator"
    / "snapshots"
)

# Standard ASL Alphabet: A through Z (26 classes)
CLASSES = [chr(65 + i) for i in range(26)]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SignLanguageResNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def find_model_path() -> str | None:
    if not MODEL_SNAPSHOT_DIR.exists():
        return None
    for root, _, files in os.walk(MODEL_SNAPSHOT_DIR):
        if "best_model.pth" in files:
            return os.path.join(root, "best_model.pth")
    return None


@lru_cache(maxsize=1)
def get_classifier():
    model_path = find_model_path()
    if not model_path:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="huzaifanasirrr/realtime-sign-language-translator",
            filename="best_model.pth",
        )

    device = torch.device("cpu")
    model = SignLanguageResNet(num_classes=26)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def is_available() -> bool:
    try:
        return find_model_path() is not None or get_classifier.cache_info().currsize > 0
    except Exception:
        return False


def classify(image_bytes: bytes, top_k: int = 3) -> list[dict]:
    """Classifies an ASL hand crop image. Returns top_k predictions sorted best-first."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Not a readable image") from e

    model = get_classifier()
    tensor = PREPROCESS(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, min(top_k, len(CLASSES)))

    results = []
    for score, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append({"label": CLASSES[idx], "score": float(score)})

    return results
