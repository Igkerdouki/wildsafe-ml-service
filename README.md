# WildSafe ML Service

Real-time wildlife detection and human safety monitoring.
Hosted at https://wildsafe-ml-service.onrender.com/

## Overview

WildSafe is an ML-powered detection service with two core capabilities:

1. **Wildlife Detection** - Identifies animals in video streams for vehicle-mounted cameras to alert drivers of wildlife on or near roads

2. **Human Safety Monitoring** - Detects anomalies and distress situations involving people (e.g., fallen individuals, unconscious persons, people in distress outside stores) using pose estimation

## Detection Targets

**Wildlife (12 species):**
- Deer, Elk, Moose
- Bear, Coyote, Fox
- Raccoon, Opossum, Skunk
- Goat, Horse, Wild Boar

**Human Behavior States:**
- Normal (standing, walking)
- Fallen (lying on ground)
- Distress (hunched, slumped)

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
web services. A 512 MB Render instance is not enough for local Torch + CLIP
inference. The app detects that limit and keeps the service alive, but inference
returns `503 local_ml_unavailable` until the service runs on a larger instance.
Use `PRELOAD_MODEL=blocking` only on an instance with enough memory to load the
model before serving inference requests.

## Render

This repo includes `render.yaml` for a Docker web service. Render will provide
`PORT`; the Docker command uses it automatically.

Recommended Render environment variables:

| Variable | Description |
|----------|-------------|
| `ORCHESTRATOR_ALERT_URL` | Orchestrator `/alert` endpoint |
| `PRELOAD_MODEL` | `false` to lazy-load, `background` to load after boot, `blocking` to load before serving on larger instances |
| `CLIP_MODEL_NAME` | Hugging Face CLIP model id; defaults to `openai/clip-vit-base-patch32` for lower memory usage |
| `LOCAL_ML_ENABLED` | Set `false` to intentionally run only health/WebRTC/stream endpoints without local inference |
| `LOCAL_CLIP_MIN_MEMORY_MB` | Minimum memory required before loading local CLIP; defaults to `1536` |
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
  "confidence_threshold": 0.1,
  "camera_id": "rpi-roadside-001",
  "latitude": 37.7749,
  "longitude": -122.4194,
  "road_name": "CA-1",
  "direction": "northbound",
  "mile_marker": "12.4"
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

When a processed WebRTC frame crosses the anomaly threshold, the ML service
POSTs an incident payload to `http://localhost:8090/alert`.

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

- **Classification:** CLIP ViT-L/14 zero-shot (wildlife + person detection)
- **Pose Estimation:** MediaPipe PoseLandmarker (human behavior analysis)

## Performance Results

Detection accuracy after prompt engineering optimization:

| Species | Confidence |
|---------|------------|
| Deer | 98% |
| Bear | 99% |
| Raccoon | 98% |
| Coyote | 98% |
| Moose | 95% |
| Fox | 96% |
| Elk | 95% |
| Wild Boar | 94% |

| Human State | Confidence |
|-------------|------------|
| Normal (standing/walking) | 95% |
| Fallen (lying down) | 83% |
| Distress (hunched) | 85% |

**Latency:** ~500ms per frame (Apple M4 Pro)

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

Python, FastAPI, PyTorch, CLIP, MediaPipe

## Goals

1. **Road Safety** - Reduce wildlife-vehicle collisions through real-time intelligent roadside alerts
2. **Public Safety** - Monitor for human distress situations (fallen persons, medical emergencies) in retail, public spaces, and urban environments
