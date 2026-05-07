# WildSafe ML Service

Real-time wildlife detection and human safety monitoring.

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
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app.main:app --reload

# Test health endpoint
curl http://localhost:8000/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed status with model info |
| `/predict` | POST | Detect in uploaded image (multipart) |
| `/predict/base64` | POST | Detect in base64 image |
| `/predict/url` | POST | Detect in image from URL |
| `/predict/video` | POST | Detect in uploaded video |

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
