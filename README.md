# WildSafe ML Service

Real-time wildlife and person detection for road safety applications.

## Overview

WildSafe is an ML-powered detection service that identifies wildlife and people in video streams. Designed for vehicle-mounted cameras to alert drivers of animals on or near roads.

## Target Species

- Deer, Elk, Moose
- Bear, Coyote, Fox
- Raccoon, Opossum, Skunk
- Goat, Horse, Wild Boar
- Person

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

- **Detection:** MegaDetector V6 (animal detection)
- **Classification:** ResNet50 species classifier (training)
- **Fallback:** YOLO-World (open vocabulary)

## Planned Features

- Camera + ML animal detection
- Sensor fusion (motion/proximity)
- LED roadside warning signs
- Google Maps driver alerts
- Predictive collision risk zones
- Night-time optimization
- Incident recording
- Human detection safety mode

## Tech Stack

Python, FastAPI, PyTorch, YOLO, MegaDetector

## Goal

Reduce wildlife-vehicle collisions through real-time intelligent roadside alerts.
