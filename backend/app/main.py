import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.services import hf_classifier

MAX_BYTES = 2_000_000


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the model at startup so the first camera frame is not slow.
    if not os.getenv("SIGNBRIDGE_SKIP_WARMUP"):
        hf_classifier.get_pipeline()
    yield


app = FastAPI(title="SignBridge API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "model": hf_classifier.MODEL_ID, "loaded": hf_classifier.is_loaded()}


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
