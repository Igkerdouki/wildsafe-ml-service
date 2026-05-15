import base64
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpReceiver,
    RTCSessionDescription,
)
from aiortc.rtcicetransport import parse_stun_turn_uri

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
STREAM_FPS = float(os.getenv("STREAM_FPS", "10"))
ANIMAL_CLASSES = set(DETECTION_CLASSES[:12])
PERSON_CLASSES = {
    "person_normal",
    "person_abnormal",
    "person_fallen",
    "person_distress",
}

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
ANSI_GREEN = "\033[32m"
ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"


def _summarize_ice_url(url: str) -> str:
    parsed = urlparse(url)
    transport = parse_qs(parsed.query).get("transport", ["default"])[0]
    host_port = parsed.netloc or parsed.path
    return f"{parsed.scheme}:{host_port} transport={transport}"


def _summarize_ice_servers(servers: list[RTCIceServer]) -> list[dict]:
    summaries = []
    for server in servers:
        urls = server.urls if isinstance(server.urls, list) else [server.urls]
        summaries.append(
            {
                "urls": [_summarize_ice_url(url) for url in urls],
                "username": "set" if server.username else "unset",
                "credential": "set" if server.credential else "unset",
            }
        )
    return summaries


def _summarize_sdp(sdp: str) -> dict:
    lines = sdp.splitlines()
    return {
        "bytes": len(sdp),
        "lines": len(lines),
        "candidates": sum(1 for line in lines if line.startswith("a=candidate:")),
        "media": [line for line in lines if line.startswith("m=")],
    }


def _summarize_stats_report(report) -> dict:
    reports = list(report.values()) if hasattr(report, "values") else []
    summary = {
        "report_count": len(reports),
        "types": {},
        "candidate_pairs": [],
        "local_candidates": [],
        "remote_candidates": [],
        "inbound_rtp": [],
        "transport": [],
    }
    candidates_by_id = {}

    for item in reports:
        report_type = getattr(item, "type", "unknown")
        summary["types"][report_type] = summary["types"].get(report_type, 0) + 1

        if report_type in {"local-candidate", "remote-candidate"}:
            candidate = {
                "id": getattr(item, "id", None),
                "address": getattr(item, "address", None)
                or getattr(item, "ip", None),
                "port": getattr(item, "port", None),
                "protocol": getattr(item, "protocol", None),
                "candidate_type": getattr(item, "candidateType", None),
                "priority": getattr(item, "priority", None),
                "url": (
                    _summarize_ice_url(getattr(item, "url"))
                    if getattr(item, "url", None)
                    else None
                ),
            }
            candidates_by_id[candidate["id"]] = candidate
            if report_type == "local-candidate":
                summary["local_candidates"].append(candidate)
            else:
                summary["remote_candidates"].append(candidate)

    for item in reports:
        report_type = getattr(item, "type", "unknown")
        if report_type == "candidate-pair":
            local_candidate_id = getattr(item, "localCandidateId", None)
            remote_candidate_id = getattr(item, "remoteCandidateId", None)
            summary["candidate_pairs"].append(
                {
                    "id": getattr(item, "id", None),
                    "state": getattr(item, "state", None),
                    "nominated": getattr(item, "nominated", None),
                    "current_round_trip_time": getattr(
                        item,
                        "currentRoundTripTime",
                        None,
                    ),
                    "bytes_sent": getattr(item, "bytesSent", None),
                    "bytes_received": getattr(item, "bytesReceived", None),
                    "requests_sent": getattr(item, "requestsSent", None),
                    "responses_received": getattr(item, "responsesReceived", None),
                    "local": candidates_by_id.get(local_candidate_id)
                    or local_candidate_id,
                    "remote": candidates_by_id.get(remote_candidate_id)
                    or remote_candidate_id,
                }
            )
        elif report_type == "inbound-rtp":
            summary["inbound_rtp"].append(
                {
                    "kind": getattr(item, "kind", None),
                    "packets_received": getattr(item, "packetsReceived", None),
                    "packets_lost": getattr(item, "packetsLost", None),
                    "jitter": getattr(item, "jitter", None),
                    "bytes_received": getattr(item, "bytesReceived", None),
                    "frames_decoded": getattr(item, "framesDecoded", None),
                    "frames_dropped": getattr(item, "framesDropped", None),
                }
            )
        elif report_type == "transport":
            summary["transport"].append(
                {
                    "dtls_state": getattr(item, "dtlsState", None),
                    "ice_role": getattr(item, "iceRole", None),
                    "selected_candidate_pair_id": getattr(
                        item,
                        "selectedCandidatePairId",
                        None,
                    ),
                }
            )

    return summary


async def _log_webrtc_stats(
    peer_connection: RTCPeerConnection,
    stream_id: str,
    camera_id: str,
    reason: str,
):
    try:
        report = await peer_connection.getStats()
    except Exception as exc:
        logger.warning(
            "WebRTC stats unavailable stream_id=%s camera_id=%s reason=%s error=%r",
            stream_id,
            camera_id,
            reason,
            exc,
        )
        return

    logger.info(
        "WebRTC stats stream_id=%s camera_id=%s reason=%s summary=%s",
        stream_id,
        camera_id,
        reason,
        _summarize_stats_report(report),
    )


def _normalize_ice_urls(server: dict) -> list[str]:
    raw_urls = server.get("urls", server.get("url"))
    if raw_urls is None:
        raise ValueError("ICE server entry must include 'urls' or 'url'")

    urls = raw_urls if isinstance(raw_urls, list) else [raw_urls]
    normalized_urls = []
    for url in urls:
        if not isinstance(url, str):
            raise ValueError("ICE server urls must be strings")

        if url.startswith("stun:") and "?transport=" in url:
            url = url.split("?transport=", 1)[0]

        parse_stun_turn_uri(url)
        normalized_urls.append(url)

    return normalized_urls


def _load_ice_servers() -> list[RTCIceServer]:
    raw_config = os.getenv("WEBRTC_ICE_SERVERS")
    if not raw_config:
        return []

    servers = json.loads(raw_config)
    if not isinstance(servers, list):
        raise ValueError("WEBRTC_ICE_SERVERS must be a JSON array")

    return [
        RTCIceServer(
            urls=_normalize_ice_urls(server),
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
    frames_skipped: int = 0
    frames_processed: int = 0
    stream_frames_encoded: int = 0
    latest_frame_jpeg: Optional[bytes] = None
    latest_frame_sequence: int = 0
    latest_frame_at: Optional[str] = None
    last_stream_frame_at: float = 0.0
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


def _format_prediction_for_terminal(result: dict) -> str:
    predicted_label = result["predicted_species"]
    confidence = result["confidence"]

    if predicted_label in ANIMAL_CLASSES:
        animal_name = predicted_label.replace("_", " ")
        return f"{ANSI_GREEN}ANIMAL{ANSI_RESET} {animal_name} confidence={confidence:.4f}"

    if predicted_label in PERSON_CLASSES:
        person_state = predicted_label.replace("person_", "").replace("_", " ")
        pose = result.get("pose_analysis")
        if pose:
            pose_behavior = pose.get("pose_behavior") or "unknown_pose"
            body_angle = pose.get("body_angle")
            return (
                f"{ANSI_BLUE}PERSON{ANSI_RESET} {person_state} "
                f"doing={pose_behavior} body_angle={body_angle} "
                f"confidence={confidence:.4f}"
            )
        return f"{ANSI_BLUE}PERSON{ANSI_RESET} {person_state} confidence={confidence:.4f}"

    return f"UNKNOWN {predicted_label} confidence={confidence:.4f}"


def _store_stream_frame(state: WebRTCStreamState, image: np.ndarray):
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return

    state.latest_frame_jpeg = encoded.tobytes()
    state.latest_frame_sequence += 1
    state.stream_frames_encoded += 1
    state.latest_frame_at = _utc_now_iso()


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
        "speaker_frequency_hz": prediction.get("speaker_frequency_hz", 0),
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
    should_report, _ = _incident_report_decision(
        state,
        incident_type,
        confidence,
        now,
    )
    return should_report


def _incident_report_decision(
    state: WebRTCStreamState,
    incident_type: str,
    confidence: float,
    now: float,
) -> tuple[bool, str]:
    if confidence < INCIDENT_CONFIDENCE_THRESHOLD:
        return False, "below_threshold"
    if state.last_incident_type != incident_type:
        return True, "new_incident_type"
    elapsed = now - state.last_incident_sent_at
    if elapsed >= INCIDENT_COOLDOWN_SECONDS:
        return True, "cooldown_elapsed"
    return False, "cooldown_active"


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
        start = time.perf_counter()
        response = await client.post(ORCHESTRATOR_ALERT_URL, json=payload)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Incident orchestrator response stream_id=%s camera_id=%s status=%s elapsed_ms=%.1f",
            stream_id,
            state.camera_id,
            response.status_code,
            elapsed_ms,
        )


@app.on_event("startup")
async def startup_event():
    """Optionally load CLIP at service startup."""
    preload_model = os.getenv("PRELOAD_MODEL", "false").lower()
    logger.info(
        "ML service startup preload_model=%s webrtc_ice_servers_present=%s orchestrator_alert_url=%s",
        preload_model,
        bool(os.getenv("WEBRTC_ICE_SERVERS")),
        ORCHESTRATOR_ALERT_URL,
    )
    if preload_model in {"blocking", "1", "true", "yes"}:
        logger.info("Blocking startup until CLIP model preload completes")
        start = time.perf_counter()
        await asyncio.to_thread(load_model)
        elapsed_s = time.perf_counter() - start
        logger.info("CLIP model preload completed during startup elapsed_s=%.2f", elapsed_s)
    elif preload_model == "background":
        logger.info("Scheduling background CLIP model preload")
        asyncio.create_task(asyncio.to_thread(load_model))
    else:
        logger.info("CLIP model preload disabled; model will load on first inference")


@app.on_event("shutdown")
async def shutdown_event():
    """Close active WebRTC peer connections on server shutdown."""
    logger.info("ML service shutdown active_webrtc_streams=%s", len(webrtc_streams))
    await asyncio.gather(
        *(state.peer_connection.close() for state in webrtc_streams.values()),
        return_exceptions=True,
    )
    webrtc_streams.clear()
    logger.info("ML service shutdown complete active_webrtc_streams=0")


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


async def _mjpeg_stream_generator(stream_id: str):
    last_sequence = 0
    boundary = b"--frame\r\n"

    while True:
        state = webrtc_streams.get(stream_id)
        if not state:
            break

        if state.latest_frame_jpeg and state.latest_frame_sequence != last_sequence:
            last_sequence = state.latest_frame_sequence
            frame = state.latest_frame_jpeg
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + f"X-Frame-Sequence: {last_sequence}\r\n".encode()
                + f"X-Camera-Id: {state.camera_id}\r\n".encode()
                + b"\r\n"
                + frame
                + b"\r\n"
            )
        else:
            await asyncio.sleep(0.1)


def _stream_response(stream_id: str) -> StreamingResponse:
    if stream_id not in webrtc_streams:
        raise HTTPException(404, "Unknown WebRTC stream")

    return StreamingResponse(
        _mjpeg_stream_generator(stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/stream")
def stream_latest():
    """
    View the latest active WebRTC camera feed as MJPEG.

    If more than one camera is connected, returns a small HTML page with links
    to `/stream/{stream_id}` for each active stream.
    """
    if not webrtc_streams:
        raise HTTPException(404, "No active WebRTC streams")

    if len(webrtc_streams) == 1:
        stream_id = next(iter(webrtc_streams))
        return _stream_response(stream_id)

    links = "\n".join(
        f'<li><a href="/stream/{stream_id}">{state.camera_id}</a> '
        f"({state.status}, frames={state.frames_received})</li>"
        for stream_id, state in webrtc_streams.items()
    )
    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
          <head><title>WildSafe Camera Streams</title></head>
          <body>
            <h1>Active Camera Streams</h1>
            <ul>{links}</ul>
          </body>
        </html>
        """
    )


@app.get("/stream/{stream_id}")
def stream_by_id(stream_id: str):
    """View a specific active WebRTC camera feed as MJPEG."""
    return _stream_response(stream_id)


async def _process_webrtc_video_track(
    stream_id: str,
    track,
    confidence_threshold: float,
    sample_fps: float,
    use_pose_detection: bool,
):
    state = webrtc_streams[stream_id]
    min_interval = 1.0 / sample_fps
    stream_min_interval = 1.0 / STREAM_FPS if STREAM_FPS > 0 else 0.0
    last_processed_at = 0.0

    logger.info(
        "WebRTC frame loop started stream_id=%s camera_id=%s sample_fps=%.2f min_interval_s=%.3f stream_fps=%.2f confidence_threshold=%.2f use_pose_detection=%s",
        stream_id,
        state.camera_id,
        sample_fps,
        min_interval,
        STREAM_FPS,
        confidence_threshold,
        use_pose_detection,
    )
    try:
        recv_task = asyncio.create_task(track.recv())
        while True:
            done, _ = await asyncio.wait({recv_task}, timeout=5.0)
            if not done:
                logger.warning(
                    "WebRTC waiting for first/next frame stream_id=%s camera_id=%s received=%s skipped=%s processed=%s track_ready_state=%s pc_state=%s ice_state=%s signaling_state=%s",
                    stream_id,
                    state.camera_id,
                    state.frames_received,
                    state.frames_skipped,
                    state.frames_processed,
                    getattr(track, "readyState", "unknown"),
                    state.peer_connection.connectionState,
                    state.peer_connection.iceConnectionState,
                    state.peer_connection.signalingState,
                )
                await _log_webrtc_stats(
                    state.peer_connection,
                    stream_id,
                    state.camera_id,
                    "frame_recv_waiting",
                )
                continue

            frame = recv_task.result()
            recv_task = asyncio.create_task(track.recv())
            state.frames_received += 1

            now = asyncio.get_running_loop().time()
            should_process = now - last_processed_at >= min_interval
            should_update_stream = (
                stream_min_interval == 0.0
                or now - state.last_stream_frame_at >= stream_min_interval
            )

            image = None
            if should_process or should_update_stream:
                image = frame.to_ndarray(format="bgr24")

            if should_update_stream and image is not None:
                _store_stream_frame(state, image)
                state.last_stream_frame_at = now

            if not should_process:
                state.frames_skipped += 1
                if state.frames_skipped == 1 or state.frames_skipped % 30 == 0:
                    logger.info(
                        "WebRTC frame skipped stream_id=%s camera_id=%s received=%s skipped=%s processed=%s stream_frames=%s reason=sample_interval",
                        stream_id,
                        state.camera_id,
                        state.frames_received,
                        state.frames_skipped,
                        state.frames_processed,
                        state.stream_frames_encoded,
                    )
                continue

            last_processed_at = now
            logger.info(
                "WebRTC frame selected stream_id=%s camera_id=%s received=%s skipped=%s processed=%s stream_frames=%s",
                stream_id,
                state.camera_id,
                state.frames_received,
                state.frames_skipped,
                state.frames_processed,
                state.stream_frames_encoded,
            )
            logger.info(
                "WebRTC inference started stream_id=%s camera_id=%s frame=%s width=%s height=%s",
                stream_id,
                state.camera_id,
                state.frames_processed + 1,
                image.shape[1],
                image.shape[0],
            )
            inference_start = time.perf_counter()
            result = await asyncio.to_thread(
                predict_frame,
                image,
                confidence_threshold,
                use_pose_detection,
            )
            elapsed_ms = (time.perf_counter() - inference_start) * 1000
            state.latest_prediction = result
            state.frames_processed += 1
            state.status = "running"

            predicted_label = result["predicted_species"]
            confidence = result["confidence"]
            incident_type = _incident_type_for_prediction(predicted_label)
            logger.info(
                "WebRTC prediction stream_id=%s camera_id=%s frame=%s %s incident_type=%s inference_elapsed_ms=%.1f reported_inference_ms=%s alert=%s speaker_frequency_hz=%s",
                stream_id,
                state.camera_id,
                state.frames_processed,
                _format_prediction_for_terminal(result),
                incident_type or "none",
                elapsed_ms,
                result.get("inference_time_ms"),
                result.get("alert"),
                result.get("speaker_frequency_hz"),
            )
            if incident_type:
                occurred_at = _utc_now_iso()
                should_report, reason = _incident_report_decision(
                    state,
                    incident_type,
                    confidence,
                    now,
                )
                logger.info(
                    "Incident decision stream_id=%s camera_id=%s incident_type=%s confidence=%.4f threshold=%.4f should_report=%s reason=%s",
                    stream_id,
                    state.camera_id,
                    incident_type,
                    confidence,
                    INCIDENT_CONFIDENCE_THRESHOLD,
                    should_report,
                    reason,
                )
                if should_report:
                    try:
                        await _send_incident_to_orchestrator(
                            stream_id,
                            state,
                            result,
                            incident_type,
                            occurred_at,
                        )
                        state.last_orchestrator_error = None
                        logger.info(
                            "Incident posted stream_id=%s camera_id=%s incident_type=%s destination=%s",
                            stream_id,
                            state.camera_id,
                            incident_type,
                            ORCHESTRATOR_ALERT_URL,
                        )
                    except Exception as exc:
                        state.last_orchestrator_error = str(exc)
                        logger.warning(
                            "Incident post failed stream_id=%s camera_id=%s incident_type=%s destination=%s error=%s",
                            stream_id,
                            state.camera_id,
                            incident_type,
                            ORCHESTRATOR_ALERT_URL,
                            exc,
                        )
                    finally:
                        state.last_incident_type = incident_type
                        state.last_incident_sent_at = now
                else:
                    logger.info(
                        "Incident not posted stream_id=%s camera_id=%s incident_type=%s reason=%s",
                        stream_id,
                        state.camera_id,
                        incident_type,
                        reason,
                    )
    except asyncio.CancelledError:
        logger.info("WebRTC frame loop cancelled stream_id=%s camera_id=%s", stream_id, state.camera_id)
        if "recv_task" in locals() and not recv_task.done():
            recv_task.cancel()
        raise
    except Exception as exc:
        state.status = "ended"
        state.error = str(exc)
        logger.exception(
            "WebRTC frame loop ended with error stream_id=%s camera_id=%s received=%s skipped=%s processed=%s error=%r",
            stream_id,
            state.camera_id,
            state.frames_received,
            state.frames_skipped,
            state.frames_processed,
            exc,
        )


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
    try:
        ice_servers = _load_ice_servers()
    except Exception as exc:
        logger.warning("Invalid WEBRTC_ICE_SERVERS configuration: %s", exc)
        raise HTTPException(500, f"Invalid WEBRTC_ICE_SERVERS configuration: {exc}")

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
    logger.info(
        "WebRTC offer accepted stream_id=%s camera_id=%s lat=%s lon=%s road_name=%s direction=%s mile_marker=%s sample_fps=%.2f confidence_threshold=%.2f use_pose_detection=%s ice_servers=%s offer_summary=%s",
        stream_id,
        request.camera_id,
        request.latitude,
        request.longitude,
        request.road_name,
        request.direction,
        request.mile_marker,
        request.sample_fps,
        request.confidence_threshold,
        request.use_pose_detection,
        len(ice_servers),
        _summarize_sdp(request.sdp),
    )
    logger.info(
        "WebRTC ICE servers stream_id=%s camera_id=%s summary=%s",
        stream_id,
        request.camera_id,
        _summarize_ice_servers(ice_servers),
    )

    @peer_connection.on("track")
    def on_track(track):
        logger.info(
            "WebRTC track received stream_id=%s camera_id=%s kind=%s id=%s",
            stream_id,
            state.camera_id,
            track.kind,
            getattr(track, "id", "unknown"),
        )
        if track.kind != "video":
            logger.info(
                "WebRTC non-video track ignored stream_id=%s camera_id=%s kind=%s",
                stream_id,
                state.camera_id,
                track.kind,
            )
            return

        state.status = "receiving"
        logger.info(
            "WebRTC video track received stream_id=%s camera_id=%s track_id=%s",
            stream_id,
            state.camera_id,
            getattr(track, "id", "unknown"),
        )
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
        logger.info(
            "WebRTC connection state stream_id=%s camera_id=%s state=%s ice_state=%s signaling_state=%s",
            stream_id,
            state.camera_id,
            peer_connection.connectionState,
            peer_connection.iceConnectionState,
            peer_connection.signalingState,
        )
        await _log_webrtc_stats(
            peer_connection,
            stream_id,
            state.camera_id,
            f"connection_state_{peer_connection.connectionState}",
        )
        if peer_connection.connectionState in {"failed", "closed", "disconnected"}:
            logger.info(
                "Closing WebRTC peer connection stream_id=%s camera_id=%s reason=connection_state_%s",
                stream_id,
                state.camera_id,
                peer_connection.connectionState,
            )
            await peer_connection.close()

    @peer_connection.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(
            "WebRTC ICE connection state stream_id=%s camera_id=%s state=%s connection_state=%s signaling_state=%s",
            stream_id,
            state.camera_id,
            peer_connection.iceConnectionState,
            peer_connection.connectionState,
            peer_connection.signalingState,
        )
        await _log_webrtc_stats(
            peer_connection,
            stream_id,
            state.camera_id,
            f"ice_connection_state_{peer_connection.iceConnectionState}",
        )

    @peer_connection.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(
            "WebRTC ICE gathering state stream_id=%s camera_id=%s state=%s local_description=%s",
            stream_id,
            state.camera_id,
            peer_connection.iceGatheringState,
            (
                _summarize_sdp(peer_connection.localDescription.sdp)
                if peer_connection.localDescription
                else None
            ),
        )
        if peer_connection.iceGatheringState == "complete":
            await _log_webrtc_stats(
                peer_connection,
                stream_id,
                state.camera_id,
                "ice_gathering_complete",
            )

    @peer_connection.on("signalingstatechange")
    async def on_signalingstatechange():
        logger.info(
            "WebRTC signaling state stream_id=%s camera_id=%s state=%s",
            stream_id,
            state.camera_id,
            peer_connection.signalingState,
        )

    try:
        offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
        logger.info("Setting remote description stream_id=%s camera_id=%s", stream_id, state.camera_id)
        await peer_connection.setRemoteDescription(offer)
        logger.info("Remote description set stream_id=%s camera_id=%s", stream_id, state.camera_id)
        await _log_webrtc_stats(peer_connection, stream_id, state.camera_id, "remote_description_set")
        _prefer_h264(peer_connection)
        logger.info("Codec preferences applied stream_id=%s camera_id=%s preferred=h264", stream_id, state.camera_id)
        answer = await peer_connection.createAnswer()
        logger.info("WebRTC answer created stream_id=%s camera_id=%s summary=%s", stream_id, state.camera_id, _summarize_sdp(answer.sdp))
        await peer_connection.setLocalDescription(answer)
        logger.info(
            "Local description set stream_id=%s camera_id=%s type=%s summary=%s",
            stream_id,
            state.camera_id,
            peer_connection.localDescription.type,
            _summarize_sdp(peer_connection.localDescription.sdp),
        )
        await _log_webrtc_stats(peer_connection, stream_id, state.camera_id, "local_description_set")
    except Exception as exc:
        logger.exception("Could not create WebRTC answer stream_id=%s camera_id=%s error=%r", stream_id, state.camera_id, exc)
        await peer_connection.close()
        webrtc_streams.pop(stream_id, None)
        logger.info("Removed WebRTC stream stream_id=%s reason=answer_error", stream_id)
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
        stream_frames_encoded=state.stream_frames_encoded,
        latest_frame_at=state.latest_frame_at,
        stream_url=f"/stream/{stream_id}",
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

    logger.info("Closing WebRTC stream by API stream_id=%s camera_id=%s", stream_id, state.camera_id)
    await state.peer_connection.close()
    logger.info("Removed WebRTC stream stream_id=%s reason=delete_api", stream_id)
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
