"""Runs a pretrained Hugging Face image-classification model locally.

The model is downloaded once (about 343 MB for the default) into the Hugging Face
cache and then runs fully on your own machine.
Switch models with:  SIGNBRIDGE_HF_MODEL=owner/model-name
"""
import io
import os
from functools import lru_cache

from PIL import Image, UnidentifiedImageError

MODEL_ID = os.getenv("SIGNBRIDGE_HF_MODEL", "Hemgg/sign-language-classification")


@lru_cache(maxsize=1)
def get_pipeline():
    # Imported lazily so the API (and its tests) start without loading PyTorch.
    from transformers import pipeline

    # device=-1 is CPU. For an NVIDIA GPU use device=0.
    return pipeline("image-classification", model=MODEL_ID, device=-1)


def is_loaded() -> bool:
    return get_pipeline.cache_info().currsize > 0


def classify(image_bytes: bytes, top_k: int = 3) -> list[dict]:
    """Returns [{"label": "A", "score": 0.97}, ...] sorted best-first."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Not a readable image") from e
    results = get_pipeline()(image, top_k=top_k)
    return [{"label": str(r["label"]), "score": float(r["score"])} for r in results]
