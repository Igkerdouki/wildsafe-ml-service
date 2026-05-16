# WildSafe ML Service

Real-time roadside animal detection and human security monitoring.
Hosted at https://wildsafe-ml-service.onrender.com/

## Overview

WildSafe is an ML-powered road-safety detection service with two core capabilities:

1. **Animal Roadway Detection** - Detects animals only when they enter the configured roadway danger zone

2. **Human Security Monitoring** - Detects risky people-related situations in the roadway danger zone, such as lingering, fallen/lying posture, erratic movement, or close multi-person interaction

## Detection Targets

**Animal road incidents:**
- Generic COCO animal detections in the configured road ROI
- Species-specific wildlife detection is a later phase that needs a trained wildlife ONNX model

**Human road/security incidents:**
- Fallen/lying in road ROI
- Lingering in road ROI
- Erratic movement in road ROI
- Close multi-person interaction in road ROI

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
fastapi dev app/main.py

# Test health endpoint
curl http://localhost:8000/health
```

## Docker

```bash
docker build -t wildsafe-ml-service .
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e ORCHESTRATOR_ALERT_URL=https://smart-wild.onrender.com/alert \
  wildsafe-ml-service
```

The container starts Uvicorn on `0.0.0.0:$PORT`, which is required by Render
web services. The production detector uses OpenCV DNN with a YOLO ONNX model
to reduce memory use on small instances.

## Render

This repo includes `render.yaml` for a Docker web service. Render will provide
`PORT`; the Docker command uses it automatically.

Recommended Render environment variables:

| Variable | Description |
|----------|-------------|
| `ORCHESTRATOR_ALERT_URL` | Orchestrator `/alert` endpoint |
| `PRELOAD_MODEL` | `false` to lazy-load, `background` to load after boot, `blocking` to load before serving |
| `DETECTOR_MODEL_PATH` | ONNX detector path; defaults to `/app/models/yolo11n.onnx` in Docker |
| `DETECTOR_MODEL_URL` | URL used to download the ONNX detector if missing |
| `DETECTION_CONFIDENCE_THRESHOLD` | Detector confidence threshold; defaults to `0.35` |
| `INCIDENT_CONFIDENCE_THRESHOLD` | Minimum confidence before posting incidents; defaults to `0.35` |
| `PERSON_LINGER_SECONDS` | Seconds a person must remain in road ROI before lingering alert; defaults to `3.0` |
| `PERSON_ERRATIC_WINDOW_SECONDS` | Time window for erratic movement analysis; defaults to `5.0` |
| `WEBRTC_ICE_SERVERS` | Optional JSON array of STUN/TURN servers for WebRTC |

Example `WEBRTC_ICE_SERVERS`:

```json
[
  {
    "urls": ["turns:turn.example.com:443"],
    "username": "user",
    "credential": "password"
  }
]
```

Render web services expose HTTP through Render's load balancer. WebRTC media
usually needs UDP connectivity, so production WebRTC on Render should use a TURN
relay configured through `WEBRTC_ICE_SERVERS`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed status with model info |
| `/predict` | POST | Detect in uploaded image (multipart) |
| `/predict/base64` | POST | Detect in base64 image |
| `/predict/url` | POST | Detect in image from URL |
| `/predict/video` | POST | Detect in uploaded video |
| `/predict/webrtc/offer` | POST | Start live detection from a WebRTC H.264 video stream |
| `/predict/webrtc/{stream_id}` | GET | Get latest prediction for a live WebRTC stream |
| `/stream` | GET | View active WebRTC camera streams |
| `/stream/{stream_id}` | GET | View a specific WebRTC camera feed page |
| `/stream/{stream_id}/mjpeg` | GET | View a specific WebRTC camera feed as raw MJPEG |

## WebRTC Streaming

For a Raspberry Pi camera stream, send an SDP offer to:

```bash
POST /predict/webrtc/offer
```

Request body:

```json
{
  "sdp": "v=0...",
  "type": "offer",
  "sample_fps": 3.0,
  "confidence_threshold": 0.35,
  "camera_id": "rpi-roadside-001",
  "latitude": 37.7749,
  "longitude": -122.4194,
  "road_name": "CA-1",
  "direction": "northbound",
  "mile_marker": "12.4",
  "road_roi": [
    {"x": 0.20, "y": 0.55},
    {"x": 0.80, "y": 0.55},
    {"x": 1.00, "y": 1.00},
    {"x": 0.00, "y": 1.00}
  ]
}
```

The response contains a WebRTC SDP answer and a `stream_id`. Set the answer as
the Pi client's remote description, then poll:

```bash
GET /predict/webrtc/{stream_id}
```

To view the received camera footage in a browser, open:

```bash
GET /stream/{stream_id}
```

If only one camera is connected, `/stream` opens that feed viewer. If multiple
cameras are connected, `/stream` lists links for each active stream. The raw
MJPEG feed is available at `/stream/{stream_id}/mjpeg`.

Close a stream with:

```bash
DELETE /predict/webrtc/{stream_id}
```

When a processed WebRTC frame produces an animal or person security incident in
the configured `road_roi`, the ML service POSTs an incident payload to the
configured `ORCHESTRATOR_ALERT_URL`. If `road_roi` is missing, the service still
streams and reports predictions but does not post incidents.

## Project Structure

```
wildsafe-ml-service/
├── app/                      # FastAPI application
│   ├── main.py              # API endpoints
│   ├── inference.py         # Model loading & prediction
│   └── schemas.py           # Pydantic models
│
├── research/                 # Research & experiments
│   ├── benchmark.py         # Benchmark script
│   ├── train_*.py           # Training scripts
│   ├── evaluate_*.py        # Evaluation scripts
│   └── datasets/            # Training/test data splits
│
├── models/                   # Trained model weights
└── requirements.txt          # Python dependencies
```

## Current Models

- **Object detector:** YOLO ONNX through OpenCV DNN
- **Road-safety logic:** rule-based road ROI, animal, and person security heuristics

## Planned Features

- Camera + ML animal detection
- Sensor fusion (motion/proximity)
- LED roadside warning signs
- Google Maps driver alerts
- Predictive collision risk zones
- Night-time optimization
- Incident recording
- Real-time anomaly alerts for businesses/public spaces

## Tech Stack

Python, FastAPI, OpenCV DNN, ONNX, aiortc

## Goals

1. **Road Safety** - Reduce wildlife-vehicle collisions through real-time intelligent roadside alerts
2. **Public Safety** - Monitor for roadway security situations involving people in unsafe road zones
