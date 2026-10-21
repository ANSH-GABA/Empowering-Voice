"""Inference for the MLP artifact exported by Model/scripts/train_static.py."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

FEATURE_DIM = 129
MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "frontend"
    / "public"
    / "models"
    / "static_mlp.json"
)


@dataclass(frozen=True)
class StaticMlpModel:
    classes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]


def is_available() -> bool:
    return MODEL_PATH.is_file()


@lru_cache(maxsize=1)
def get_model() -> StaticMlpModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained landmark model not found at {MODEL_PATH}. "
            "Run Model/scripts/train_static.py first."
        )

    try:
        payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read landmark model at {MODEL_PATH}.") from exc

    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported landmark model format.")
    if payload.get("featureDim") != FEATURE_DIM:
        raise ValueError(f"Landmark model must have {FEATURE_DIM} input features.")

    classes = payload.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(label, str) or not label for label in classes)
    ):
        raise ValueError("Landmark model must contain non-empty class labels.")

    mean = np.asarray(payload.get("mean"), dtype=np.float64)
    scale = np.asarray(payload.get("scale"), dtype=np.float64)
    if (
        mean.shape != (FEATURE_DIM,)
        or scale.shape != (FEATURE_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale == 0)
    ):
        raise ValueError("Landmark model has invalid scaler values.")

    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("Landmark model must contain dense layers.")

    layers = []
    input_dim = FEATURE_DIM
    for raw_layer in raw_layers:
        if not isinstance(raw_layer, dict):
            raise ValueError("Landmark model contains an invalid dense layer.")
        weights = np.asarray(raw_layer.get("W"), dtype=np.float64)
        bias = np.asarray(raw_layer.get("b"), dtype=np.float64)
        if (
            weights.ndim != 2
            or bias.ndim != 1
            or weights.shape[0] != input_dim
            or weights.shape[1] != bias.size
            or not np.isfinite(weights).all()
            or not np.isfinite(bias).all()
        ):
            raise ValueError("Landmark model contains invalid layer dimensions or values.")
        layers.append((weights, bias))
        input_dim = bias.size

    if input_dim != len(classes):
        raise ValueError("Landmark model output dimension does not match its classes.")

    return StaticMlpModel(tuple(classes), mean, scale, tuple(layers))


def predict(features: list[float]) -> dict[str, float | str]:
    if len(features) != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features.")
    values = np.asarray(features, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Features must all be finite numbers.")

    model = get_model()
    output = (values - model.mean) / model.scale
    for index, (weights, bias) in enumerate(model.layers):
        output = output @ weights + bias
        if index < len(model.layers) - 1:
            output = np.maximum(output, 0)

    exp = np.exp(output - np.max(output))
    probabilities = exp / exp.sum()
    best = int(np.argmax(probabilities))
    return {"label": model.classes[best], "confidence": float(probabilities[best])}
