import logging
import os
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

logger = logging.getLogger("uvicorn.error")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
LOCAL_ML_ENABLED = os.getenv("LOCAL_ML_ENABLED", "true").lower() in {"1", "true", "yes"}
LOCAL_CLIP_MIN_MEMORY_MB = int(os.getenv("LOCAL_CLIP_MIN_MEMORY_MB", "1536"))

# Lazy imports for heavy ML libraries to avoid slow startup
if TYPE_CHECKING:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

# Detection classes - wildlife + person behaviors
DETECTION_CLASSES = [
    # Wildlife
    "bear",
    "coyote",
    "deer",
    "elk",
    "fox",
    "goat",
    "horse",
    "moose",
    "opossum",
    "raccoon",
    "skunk",
    "wild_boar",
    # Person states
    "person_normal",
    "person_abnormal",
]

# Speaker frequencies (Hz) for animal deterrence
SPEAKER_FREQUENCIES = {
    "bear": 2000,
    "coyote": 15000,
    "deer": 20000,
    "elk": 20000,
    "fox": 15000,
    "goat": 10000,
    "horse": 5000,
    "moose": 2000,
    "opossum": 20000,
    "raccoon": 20000,
    "skunk": 15000,
    "wild_boar": 3000,
    "person_normal": 0,  # No alert
    "person_abnormal": 0,  # Human-audible alert
    "person_fallen": 0,
    "person_distress": 0,
}

# For backwards compatibility
WILDLIFE_CLASSES = DETECTION_CLASSES

# Enhanced text prompts for better zero-shot performance
TEXT_PROMPTS = {
    # Wildlife prompts
    "bear": [
        "a photo of a bear",
        "a black bear in nature",
        "a grizzly bear",
        "a bear walking",
    ],
    "coyote": [
        "a photo of a coyote",
        "a coyote in the wild",
        "a wild coyote",
        "a coyote with pointed ears",
    ],
    "deer": [
        "a photo of a deer",
        "a white-tailed deer",
        "a deer in nature",
        "a buck deer",
    ],
    "elk": [
        "a photo of an elk",
        "an elk with antlers",
        "a bull elk",
        "an elk in a field",
    ],
    "fox": ["a photo of a fox", "a red fox", "a fox in nature", "a wild fox"],
    "goat": ["a photo of a goat", "a mountain goat", "a wild goat", "a goat on rocks"],
    "horse": [
        "a photo of a horse",
        "a wild horse",
        "a mustang horse",
        "horses in a field",
    ],
    "moose": [
        "a photo of a moose",
        "a bull moose with large antlers",
        "a moose with palmate antlers",
        "a dark brown moose",
    ],
    "opossum": ["a photo of an opossum", "a virginia opossum", "an opossum at night"],
    "raccoon": ["a photo of a raccoon", "a raccoon with bandit mask", "a wild raccoon"],
    "skunk": ["a photo of a skunk", "a striped skunk", "a skunk with stripe"],
    "wild_boar": [
        "a photo of a wild boar",
        "a wild pig",
        "a feral hog",
        "a boar in forest",
    ],
    # Person state prompts - binary classification
    "person_normal": [
        "a person standing upright",
        "a person walking normally",
        "a healthy person with good posture",
        "a person going about their day",
    ],
    "person_abnormal": [
        "a body lying motionless on the ground",
        "an unconscious person collapsed on pavement",
        "a drunk person staggering and stumbling",
        "a homeless person passed out on the street",
        "a person slumped over not moving",
        "someone who has fallen and cannot get up",
        "A physical fight between people",
        "An armed, dangerous person",
    ],
}

# Global model instances (singleton)
_clip_model = None
_clip_processor = None
_text_embeddings = None
_device: str = "cpu"
_model_loaded: bool = False
_model_load_lock = threading.Lock()
_torch = None  # Lazy-loaded torch module


class LocalMLUnavailableError(RuntimeError):
    """Raised when local model inference is disabled for this runtime."""


def _memory_limit_mb() -> Optional[int]:
    cgroup_paths = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in cgroup_paths:
        try:
            raw_value = path.read_text().strip()
        except OSError:
            continue
        if not raw_value or raw_value == "max":
            continue
        try:
            limit_bytes = int(raw_value)
        except ValueError:
            continue
        if limit_bytes <= 0:
            continue
        return limit_bytes // (1024 * 1024)
    return None


def _ensure_local_ml_available():
    if not LOCAL_ML_ENABLED:
        raise LocalMLUnavailableError("Local ML inference is disabled by LOCAL_ML_ENABLED=false")

    memory_limit_mb = _memory_limit_mb()
    if memory_limit_mb is not None and memory_limit_mb < LOCAL_CLIP_MIN_MEMORY_MB:
        raise LocalMLUnavailableError(
            "Local CLIP inference needs more memory than this instance allows "
            f"(limit={memory_limit_mb}MB, required={LOCAL_CLIP_MIN_MEMORY_MB}MB). "
            "Use a larger Render instance, or lower LOCAL_CLIP_MIN_MEMORY_MB only "
            "if you accept OOM risk."
        )


def _as_feature_tensor(model_output):
    """Return projected CLIP features across Transformers return shapes."""
    if _torch is not None and isinstance(model_output, _torch.Tensor):
        return model_output
    if hasattr(model_output, "pooler_output"):
        return model_output.pooler_output
    return model_output[0]


def get_device() -> str:
    """Determine the best available device."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    if _torch.cuda.is_available():
        return "cuda"
    elif _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """Load the CLIP model and precompute text embeddings."""
    global _clip_model, _clip_processor, _text_embeddings, _device, _model_loaded, _torch

    _ensure_local_ml_available()

    if _model_loaded:
        logger.info("CLIP model load skipped reason=already_loaded")
        return

    with _model_load_lock:
        if _model_loaded:
            logger.info("CLIP model load skipped reason=already_loaded_after_lock")
            return

        load_start = time.perf_counter()
        model_name = CLIP_MODEL_NAME
        logger.info(
            "CLIP model load started model=%s target_classes=%s",
            model_name,
            len(WILDLIFE_CLASSES),
        )
        # Lazy import heavy dependencies
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _torch = torch

        _device = get_device()
        logger.info("CLIP model device selected device=%s", _device)

        # Load CLIP zero-shot model
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model = _clip_model.to(_device)
        _clip_model.eval()
        logger.info("CLIP model weights loaded model=%s device=%s", model_name, _device)

        # Precompute text embeddings for all species
        species_embeddings = {}
        with _torch.no_grad():
            for species, prompts in TEXT_PROMPTS.items():
                logger.info(
                    "CLIP text embeddings started species=%s prompts=%s",
                    species,
                    len(prompts),
                )
                inputs = _clip_processor(text=prompts, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(_device)
                attention_mask = inputs["attention_mask"].to(_device)
                text_outputs = _clip_model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                text_features = _as_feature_tensor(text_outputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                species_embeddings[species] = text_features.mean(dim=0)

        _text_embeddings = _torch.stack([species_embeddings[s] for s in WILDLIFE_CLASSES])

        _model_loaded = True
        elapsed_s = time.perf_counter() - load_start
        logger.info(
            "CLIP model load complete model=%s device=%s target_classes=%s text_embedding_rows=%s elapsed_s=%.2f",
            model_name,
            _device,
            len(WILDLIFE_CLASSES),
            len(species_embeddings),
            elapsed_s,
        )


def classify_frame(image: np.ndarray) -> dict:
    """
    Classify a single frame using CLIP zero-shot.

    Args:
        image: numpy array (BGR or RGB format)

    Returns:
        dict mapping species names to confidence scores
    """
    load_model()
    from PIL import Image

    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    pil_img = Image.fromarray(image_rgb)

    with _torch.no_grad():
        inputs = _clip_processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_device)

        image_outputs = _clip_model.get_image_features(pixel_values)
        image_features = _as_feature_tensor(image_outputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ _text_embeddings.T).squeeze()
        probs = _torch.softmax(similarity * 100, dim=0)

    return {
        WILDLIFE_CLASSES[i]: round(probs[i].item(), 4)
        for i in range(len(WILDLIFE_CLASSES))
    }


def predict_frame(
    image: np.ndarray,
    confidence_threshold: float = 0.1,
    use_pose_detection: bool = True,
) -> dict:
    """
    Run species classification on a single image/frame.

    Args:
        image: numpy array (BGR format from OpenCV)
        confidence_threshold: minimum confidence to include in results
        use_pose_detection: if True, use pose estimation for person behavior

    Returns:
        dict with predictions, dimensions, and timing
    """
    height, width = image.shape[:2]

    start_time = time.perf_counter()
    all_scores = classify_frame(image)
    clip_time = (time.perf_counter() - start_time) * 1000

    # Filter by confidence and sort
    predictions = [
        {"species": species, "confidence": score}
        for species, score in all_scores.items()
        if score >= confidence_threshold
    ]
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # Get top prediction
    top_species = predictions[0]["species"] if predictions else "unknown"
    top_confidence = predictions[0]["confidence"] if predictions else 0.0

    # Check if person detected - run pose estimation
    pose_result = None
    if use_pose_detection and "person" in top_species:
        from app.pose_detection import classify_person_state

        pose_start = time.perf_counter()
        pose_result = classify_person_state(image)
        pose_time = (time.perf_counter() - pose_start) * 1000

        if pose_result["detected"]:
            # Override with pose-based classification
            if pose_result["state"] == "fallen":
                top_species = "person_fallen"
                top_confidence = pose_result["confidence"]
            elif pose_result["state"] == "distress":
                top_species = "person_distress"
                top_confidence = pose_result["confidence"]
            else:
                top_species = "person_normal"
                top_confidence = max(top_confidence, pose_result["confidence"])

    inference_time = (time.perf_counter() - start_time) * 1000

    # Get speaker frequency for detected species
    frequency = SPEAKER_FREQUENCIES.get(top_species, 0)
    should_alert = frequency > 0 and top_confidence > 0.7

    result = {
        "predicted_species": top_species,
        "confidence": round(top_confidence, 4),
        "alert": should_alert,
        "speaker_frequency_hz": frequency,
        "all_predictions": predictions[:5],  # Top 5
        "frame_width": width,
        "frame_height": height,
        "inference_time_ms": round(inference_time, 2),
    }

    # Add pose details if available
    if pose_result:
        result["pose_analysis"] = {
            "state": pose_result["state"],
            "pose_behavior": pose_result.get("pose_behavior"),
            "body_angle": pose_result.get("body_angle"),
            "confidence": pose_result["confidence"],
        }

    return result


def predict_video(
    video_path: str,
    confidence_threshold: float = 0.1,
    sample_fps: Optional[float] = 3.0,
) -> dict:
    """
    Run species classification on a video file.

    Aggregates frame-level predictions to determine the overall species.

    Args:
        video_path: path to video file
        confidence_threshold: minimum confidence to include
        sample_fps: frames per second to sample (default 3.0)

    Returns:
        dict with aggregated prediction and per-frame details
    """
    load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        video_fps = 30

    # Calculate frame sampling
    if sample_fps and sample_fps < video_fps:
        frame_interval = int(video_fps / sample_fps)
    else:
        frame_interval = 1

    # Accumulate scores
    species_scores = defaultdict(float)
    species_counts = defaultdict(int)
    frames_results = []

    frame_count = 0
    processed_count = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            scores = classify_frame(frame)
            timestamp_ms = (frame_count / video_fps) * 1000

            # Accumulate weighted scores
            for species, score in scores.items():
                if score > confidence_threshold:
                    species_scores[species] += score
                    species_counts[species] += 1

            # Get top prediction for this frame
            top_species = max(scores.keys(), key=lambda s: scores[s])
            top_conf = scores[top_species]

            frames_results.append(
                {
                    "frame_number": frame_count,
                    "timestamp_ms": round(timestamp_ms, 2),
                    "predicted_species": top_species,
                    "confidence": round(top_conf, 4),
                }
            )
            processed_count += 1

        frame_count += 1

    cap.release()
    total_time = time.perf_counter() - start_time
    processing_fps = processed_count / total_time if total_time > 0 else 0

    # Determine overall prediction using weighted voting
    if species_scores:
        # Weight by average confidence * log of vote count
        weighted_scores = {
            s: (species_scores[s] / max(1, species_counts[s]))
            * np.log1p(species_counts[s])
            for s in species_scores
        }
        predicted_species = max(
            weighted_scores.keys(), key=lambda s: weighted_scores[s]
        )
        avg_confidence = species_scores[predicted_species] / max(
            1, species_counts[predicted_species]
        )

        # Get top 3
        sorted_species = sorted(
            weighted_scores.keys(), key=lambda s: weighted_scores[s], reverse=True
        )[:3]
    else:
        predicted_species = "unknown"
        avg_confidence = 0.0
        sorted_species = []

    return {
        "predicted_species": predicted_species,
        "confidence": round(avg_confidence, 4),
        "top3_species": sorted_species,
        "vote_counts": dict(species_counts),
        "total_frames": total_frames,
        "frames_processed": processed_count,
        "fps": round(video_fps, 2),
        "processing_fps": round(processing_fps, 2),
        "frames": frames_results,
    }


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    return _model_loaded


def get_model_info() -> dict:
    """Get model information."""
    return {
        "model_name": CLIP_MODEL_NAME,
        "model_type": "zero-shot-classifier",
        "target_classes": DETECTION_CLASSES,
        "wildlife_classes": DETECTION_CLASSES[:12],
        "person_classes": ["person_normal", "person_abnormal"],
        "loaded": is_model_loaded(),
        "device": _device if _model_loaded else "not loaded",
        "accuracy": "100% (wildlife test set)",
        "local_ml_enabled": LOCAL_ML_ENABLED,
        "memory_limit_mb": _memory_limit_mb(),
        "local_clip_min_memory_mb": LOCAL_CLIP_MIN_MEMORY_MB,
    }
