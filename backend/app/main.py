import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.services import asl_classifier, hf_classifier, static_classifier
from core.config import get_settings
from app.api.deps import SettingsDep

MAX_BYTES = 2_000_000
LANDMARK_FEATURE_DIM = 129


class LandmarkPredictionRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: list[float]) -> list[float]:
        if len(features) != LANDMARK_FEATURE_DIM:
            raise ValueError(f"Expected {LANDMARK_FEATURE_DIM} features.")
        if not all(math.isfinite(value) for value in features):
            raise ValueError("Features must all be finite numbers.")
        return features


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if not settings.skip_warmup:
        if asl_classifier.is_available():
            asl_classifier.get_classifier()
        elif settings.recognizer == "hf":
            hf_classifier.get_pipeline()
    if static_classifier.is_available():
        static_classifier.get_model()
    yield


app = FastAPI(title="Empowering Voice API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health(settings: SettingsDep):
    active_model = "asl_resnet18" if asl_classifier.is_available() else hf_classifier.MODEL_ID
    return {
        "ok": True,
        "model": active_model,
        "loaded": True,
        "asl_model_available": asl_classifier.is_available(),
        "landmark_model_available": static_classifier.is_available(),
        "app": settings.app_name,
        "environment": settings.environment,
    }


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 2 MB)")
    try:
        if asl_classifier.is_available():
            predictions = asl_classifier.classify(data)
            model_name = "asl_resnet18 (99.7% ASL)"
        else:
            predictions = hf_classifier.classify(data)
            model_name = hf_classifier.MODEL_ID
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"model": model_name, "predictions": predictions}


@app.post("/api/landmark/predict")
def predict_landmarks(request: LandmarkPredictionRequest):
    try:
        prediction = static_classifier.predict(request.features)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    return {"model": "static_mlp", **prediction}


class DatasetUploadRequest(BaseModel):
    participant: str
    csv_content: str


@app.post("/api/dataset/save")
def save_dataset(request: DatasetUploadRequest):
    import datetime
    from pathlib import Path

    data_dir = Path(__file__).resolve().parents[2] / "Model" / "ml" / "data" / "processed" / "static"
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_p = "".join(c for c in request.participant if c.isalnum() or c in ("-", "_")) or "P01"
    filepath = data_dir / f"{clean_p}_{stamp}.csv"
    filepath.write_text(request.csv_content, encoding="utf-8")
    return {"ok": True, "file": filepath.name}


@app.post("/api/train/run")
def run_training():
    import subprocess
    import sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "Model" / "scripts" / "train_static.py"
    res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    if res.returncode != 0:
        raise HTTPException(500, f"Training failed: {res.stderr or res.stdout}")
    static_classifier.get_model.cache_clear()
    return {"ok": True, "output": res.stdout}