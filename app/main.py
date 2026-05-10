import base64
import asyncio
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpReceiver,
    RTCSessionDescription,
)

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
    DETECTION_CLASSES,
    predict_frame,
    predict_video,
    load_model,
    get_model_info,
)

ORCHESTRATOR_ALERT_URL = os.getenv(
    "ORCHESTRATOR_ALERT_URL",
    "https://smart-wild.onrender.com/alert",
)
INCIDENT_CONFIDENCE_THRESHOLD = 0.7
INCIDENT_COOLDOWN_SECONDS = 30.0
ANIMAL_CLASSES = set(DETECTION_CLASSES[:12])
PERSON_CLASSES = {
    "person_normal",
    "person_abnormal",
    "person_fallen",
    "person_distress",
}


def _load_ice_servers() -> list[RTCIceServer]:
    raw_config = os.getenv("WEBRTC_ICE_SERVERS")
    if not raw_config:
        return []

    servers = json.loads(raw_config)
    return [
        RTCIceServer(
            urls=server["urls"],
            username=server.get("username"),
            credential=server.get("credential"),
        )
        for server in servers
    ]

app = FastAPI(
    title="WildSafe ML Service",
    description="Wildlife species classification API using CLIP zero-shot",
    version="0.2.0"
)


@dataclass
class WebRTCStreamState:
    peer_connection: RTCPeerConnection
    camera_id: str
    latitude: float
    longitude: float
    road_name: Optional[str] = None
    direction: Optional[str] = None
    mile_marker: Optional[str] = None
    status: str = "connecting"
    frames_received: int = 0
    frames_processed: int = 0
    latest_prediction: Optional[dict] = None
    error: Optional[str] = None
    last_incident_type: Optional[str] = None
    last_incident_sent_at: float = 0.0
    last_orchestrator_error: Optional[str] = None


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _incident_type_for_prediction(predicted_label: str) -> Optional[str]:
    if predicted_label in ANIMAL_CLASSES:
        return "animal_on_road"
    if predicted_label in PERSON_CLASSES:
        return "person_on_road"
    return None


def _recommended_action(incident_type: str) -> dict:
    if incident_type == "animal_on_road":
        return {
            "priority": "high",
            "message": (
                "Animal detected on or near roadway. Trigger roadside warning "
                "and notify nearby drivers."
            ),
        }
    if incident_type == "person_on_road":
        return {
            "priority": "critical",
            "message": (
                "Person detected on or near roadway. Trigger roadside warning "
                "and notify operators for review."
            ),
        }
    return {
        "priority": "medium",
        "message": "Roadway anomaly detected. Review camera stream.",
    }


def _build_incident_payload(
    stream_id: str,
    state: WebRTCStreamState,
    prediction: dict,
    incident_type: str,
    occurred_at: str,
) -> dict:
    incident_id = f"inc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    return {
        "incident_id": incident_id,
        "type": incident_type,
        "occurred_at": occurred_at,
        "reported_at": _utc_now_iso(),
        "location": {
            "latitude": state.latitude,
            "longitude": state.longitude,
            "road_name": state.road_name,
            "direction": state.direction,
            "mile_marker": state.mile_marker,
            "camera_id": state.camera_id,
        },
        "recommended_action": _recommended_action(incident_type),
        "evidence": {
            "snapshot_url": None,
            "video_clip_url": None,
        },
    }


def _should_report_incident(
    state: WebRTCStreamState,
    incident_type: str,
    confidence: float,
    now: float,
) -> bool:
    if confidence < INCIDENT_CONFIDENCE_THRESHOLD:
        return False
    if state.last_incident_type != incident_type:
        return True
    return now - state.last_incident_sent_at >= INCIDENT_COOLDOWN_SECONDS


async def _send_incident_to_orchestrator(
    stream_id: str,
    state: WebRTCStreamState,
    prediction: dict,
    incident_type: str,
    occurred_at: str,
):
    payload = _build_incident_payload(
        stream_id=stream_id,
        state=state,
        prediction=prediction,
        incident_type=incident_type,
        occurred_at=occurred_at,
    )
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(ORCHESTRATOR_ALERT_URL, json=payload)
        response.raise_for_status()


@app.on_event("startup")
async def startup_event():
    """Start loading the CLIP model in background (disabled by default for cloud deploy)."""
    if os.getenv("PRELOAD_MODEL", "false").lower() in {"1", "true", "yes"}:
        asyncio.create_task(asyncio.to_thread(load_model))


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

            predicted_label = result["predicted_species"]
            incident_type = _incident_type_for_prediction(predicted_label)
            if incident_type:
                occurred_at = _utc_now_iso()
                confidence = result["confidence"]
                if _should_report_incident(state, incident_type, confidence, now):
                    try:
                        await _send_incident_to_orchestrator(
                            stream_id,
                            state,
                            result,
                            incident_type,
                            occurred_at,
                        )
                        state.last_orchestrator_error = None
                    except Exception as exc:
                        state.last_orchestrator_error = str(exc)
                    finally:
                        state.last_incident_type = incident_type
                        state.last_incident_sent_at = now
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
    ice_servers = _load_ice_servers()
    peer_connection = RTCPeerConnection(
        configuration=RTCConfiguration(iceServers=ice_servers)
        if ice_servers
        else None
    )
    state = WebRTCStreamState(
        peer_connection=peer_connection,
        camera_id=request.camera_id,
        latitude=request.latitude,
        longitude=request.longitude,
        road_name=request.road_name,
        direction=request.direction,
        mile_marker=request.mile_marker,
    )
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
