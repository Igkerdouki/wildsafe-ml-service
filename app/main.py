import base64
import tempfile
from typing import Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException

from app.schemas import (
    PredictionResponse,
    VideoPredictionResponse,
    Base64ImageRequest,
    URLImageRequest,
    HealthResponse,
)
from app.inference import (
    predict_frame,
    predict_video,
    load_model,
    get_model_info,
)

app = FastAPI(
    title="WildSafe ML Service",
    description="Wildlife species classification API using CLIP zero-shot",
    version="0.2.0"
)


@app.on_event("startup")
async def startup_event():
    """Pre-load the CLIP model on startup."""
    load_model()


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "WildSafe running"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Detailed health check with model status."""
    info = get_model_info()
    return HealthResponse(
        status="healthy",
        model_loaded=info["loaded"],
        model_name=info["model_name"],
        model_type=info["model_type"],
        target_classes=info["target_classes"],
        accuracy=info["accuracy"]
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_upload(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.1
):
    """
    Classify wildlife species in an uploaded image.

    Accepts: JPEG, PNG, WebP images
    Returns: Top predicted species with confidence scores
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    result = predict_frame(image, confidence_threshold)
    return PredictionResponse(**result)


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: Base64ImageRequest):
    """
    Classify wildlife species in a base64-encoded image.
    """
    try:
        image_data = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(400, "Invalid base64 encoding")

    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    result = predict_frame(image, request.confidence_threshold)
    return PredictionResponse(**result)


@app.post("/predict/url", response_model=PredictionResponse)
async def predict_url(request: URLImageRequest):
    """
    Classify wildlife species in an image fetched from URL.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(request.url, timeout=30.0)
            response.raise_for_status()
    except httpx.RequestError as e:
        raise HTTPException(400, f"Could not fetch image: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(400, f"HTTP error fetching image: {e.response.status_code}")

    nparr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image from URL")

    result = predict_frame(image, request.confidence_threshold)
    return PredictionResponse(**result)


@app.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video_upload(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.1,
    sample_fps: Optional[float] = 3.0
):
    """
    Classify wildlife species in an uploaded video.

    Aggregates frame-level predictions using weighted voting.

    Args:
        file: Video file (MP4, AVI, MOV, WebM)
        confidence_threshold: Minimum confidence to include in voting
        sample_fps: Sample rate for processing (default 3 FPS)
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")

    # Save to temp file for OpenCV
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict_video(tmp_path, confidence_threshold, sample_fps)
        return VideoPredictionResponse(**result)
    finally:
        import os
        os.unlink(tmp_path)
