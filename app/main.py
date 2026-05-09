import base64
import asyncio
import tempfile
import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from aiortc import RTCPeerConnection, RTCRtpReceiver, RTCSessionDescription

from app.schemas import (
    PredictionResponse,
    VideoPredictionResponse,
    WebRTCOfferRequest,
    WebRTCOfferResponse,
    WebRTCStreamStatus,
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


@dataclass
class WebRTCStreamState:
    peer_connection: RTCPeerConnection
    status: str = "connecting"
    frames_received: int = 0
    frames_processed: int = 0
    latest_prediction: Optional[dict] = None
    error: Optional[str] = None


webrtc_streams: dict[str, WebRTCStreamState] = {}


def _prefer_h264(peer_connection: RTCPeerConnection):
    """Prefer H.264 when the WebRTC client offers multiple video codecs."""
    video_codecs = RTCRtpReceiver.getCapabilities("video").codecs
    h264_codecs = [codec for codec in video_codecs if codec.mimeType == "video/H264"]
    if not h264_codecs:
        return

    other_codecs = [codec for codec in video_codecs if codec.mimeType != "video/H264"]
    preferred_codecs = h264_codecs + other_codecs

    for transceiver in peer_connection.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(preferred_codecs)


@app.on_event("startup")
async def startup_event():
    """Pre-load the CLIP model on startup."""
    load_model()


@app.on_event("shutdown")
async def shutdown_event():
    """Close active WebRTC peer connections on server shutdown."""
    await asyncio.gather(
        *(state.peer_connection.close() for state in webrtc_streams.values()),
        return_exceptions=True,
    )
    webrtc_streams.clear()


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


async def _process_webrtc_video_track(
    stream_id: str,
    track,
    confidence_threshold: float,
    sample_fps: float,
    use_pose_detection: bool,
):
    state = webrtc_streams[stream_id]
    min_interval = 1.0 / sample_fps
    last_processed_at = 0.0

    try:
        while True:
            frame = await track.recv()
            state.frames_received += 1

            now = asyncio.get_running_loop().time()
            if now - last_processed_at < min_interval:
                continue

            last_processed_at = now
            image = frame.to_ndarray(format="bgr24")
            result = await asyncio.to_thread(
                predict_frame,
                image,
                confidence_threshold,
                use_pose_detection,
            )
            state.latest_prediction = result
            state.frames_processed += 1
            state.status = "running"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        state.status = "ended"
        state.error = str(exc)


@app.post("/predict/webrtc/offer", response_model=WebRTCOfferResponse)
async def predict_webrtc_offer(request: WebRTCOfferRequest):
    """
    Accept a WebRTC H.264 video stream and process decoded frames in real time.

    The Raspberry Pi should POST its SDP offer here, set the returned SDP answer
    as the remote description, then poll `/predict/webrtc/{stream_id}` for the
    latest prediction.
    """
    if request.type != "offer":
        raise HTTPException(400, "WebRTC request type must be 'offer'")

    stream_id = str(uuid.uuid4())
    peer_connection = RTCPeerConnection()
    state = WebRTCStreamState(peer_connection=peer_connection)
    webrtc_streams[stream_id] = state

    @peer_connection.on("track")
    def on_track(track):
        if track.kind != "video":
            return

        state.status = "receiving"
        asyncio.create_task(
            _process_webrtc_video_track(
                stream_id,
                track,
                request.confidence_threshold,
                request.sample_fps,
                request.use_pose_detection,
            )
        )

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        state.status = peer_connection.connectionState
        if peer_connection.connectionState in {"failed", "closed", "disconnected"}:
            await peer_connection.close()

    try:
        offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
        await peer_connection.setRemoteDescription(offer)
        _prefer_h264(peer_connection)
        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)
    except Exception as exc:
        await peer_connection.close()
        webrtc_streams.pop(stream_id, None)
        raise HTTPException(400, f"Could not create WebRTC answer: {exc}")

    return WebRTCOfferResponse(
        stream_id=stream_id,
        sdp=peer_connection.localDescription.sdp,
        type=peer_connection.localDescription.type,
    )


@app.get("/predict/webrtc/{stream_id}", response_model=WebRTCStreamStatus)
def get_webrtc_prediction(stream_id: str):
    """Return the most recent prediction for an active WebRTC stream."""
    state = webrtc_streams.get(stream_id)
    if not state:
        raise HTTPException(404, "Unknown WebRTC stream")

    return WebRTCStreamStatus(
        stream_id=stream_id,
        status=state.status,
        frames_received=state.frames_received,
        frames_processed=state.frames_processed,
        latest_prediction=(
            PredictionResponse(**state.latest_prediction)
            if state.latest_prediction
            else None
        ),
        error=state.error,
    )


@app.delete("/predict/webrtc/{stream_id}")
async def close_webrtc_stream(stream_id: str):
    """Close a WebRTC stream and remove its in-memory prediction state."""
    state = webrtc_streams.pop(stream_id, None)
    if not state:
        raise HTTPException(404, "Unknown WebRTC stream")

    await state.peer_connection.close()
    return {"stream_id": stream_id, "status": "closed"}


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
