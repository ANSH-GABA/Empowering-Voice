import io
import os

os.environ["EV_SKIP_WARMUP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.main import app  # noqa: E402
from app.services import hf_classifier, static_classifier  # noqa: E402


def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, "JPEG")
    return buf.getvalue()


def test_health():
    with TestClient(app) as client:
        body = client.get("/api/health").json()
        assert body["ok"] is True and "model" in body
        assert "landmark_model_available" in body


def test_predict_returns_top_predictions(monkeypatch):
    monkeypatch.setattr(hf_classifier, "get_pipeline", lambda: lambda img, top_k: [{"label": "A", "score": 0.9}, {"label": "B", "score": 0.1}][:top_k])
    with TestClient(app) as client:
        r = client.post("/api/predict", files={"image": ("f.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["predictions"][0] == {"label": "A", "score": 0.9}


def test_predict_rejects_non_image():
    with TestClient(app) as client:
        r = client.post("/api/predict", files={"image": ("f.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_landmark_predict_uses_trained_model(monkeypatch):
    monkeypatch.setattr(
        static_classifier,
        "predict",
        lambda features: {"label": "A", "confidence": 0.9},
    )
    with TestClient(app) as client:
        r = client.post("/api/landmark/predict", json={"features": [0.0] * 129})
    assert r.status_code == 200
    assert r.json() == {
        "model": "static_mlp",
        "label": "A",
        "confidence": 0.9,
    }


def test_landmark_predict_rejects_wrong_feature_count():
    with TestClient(app) as client:
        r = client.post("/api/landmark/predict", json={"features": [0.0]})
    assert r.status_code == 422


def test_landmark_predict_reports_missing_model(monkeypatch):
    def missing_model(_features):
        raise FileNotFoundError("trained model missing")

    monkeypatch.setattr(static_classifier, "predict", missing_model)
    with TestClient(app) as client:
        r = client.post("/api/landmark/predict", json={"features": [0.0] * 129})
    assert r.status_code == 503
