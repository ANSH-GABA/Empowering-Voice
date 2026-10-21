import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.services import hf_classifier, static_classifier
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
    if not settings.skip_warmup and settings.recognizer == "hf":
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
    return {
        "ok": True,
        "model": hf_classifier.MODEL_ID,
        "loaded": hf_classifier.is_loaded(),
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
        predictions = hf_classifier.classify(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"model": hf_classifier.MODEL_ID, "predictions": predictions}


@app.post("/api/landmark/predict")
def predict_landmarks(request: LandmarkPredictionRequest):
    try:
        prediction = static_classifier.predict(request.features)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    return {"model": "static_mlp", **prediction}