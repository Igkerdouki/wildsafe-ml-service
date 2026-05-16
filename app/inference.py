import logging
import math
import os
import time
import threading
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("uvicorn.error")

DETECTOR_MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", "/app/models/yolo11n.onnx")
DETECTOR_MODEL_URL = os.getenv(
    "DETECTOR_MODEL_URL",
    "https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n.onnx",
)
DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.35"))
DETECTION_NMS_THRESHOLD = float(os.getenv("DETECTION_NMS_THRESHOLD", "0.45"))
DETECTOR_INPUT_SIZE = int(os.getenv("DETECTOR_INPUT_SIZE", "640"))
PERSON_LINGER_SECONDS = float(os.getenv("PERSON_LINGER_SECONDS", "3.0"))
PERSON_ERRATIC_WINDOW_SECONDS = float(os.getenv("PERSON_ERRATIC_WINDOW_SECONDS", "5.0"))

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
]

ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
DETECTION_CLASSES = ["animal_on_road", "person_security_issue", "no_incident"]
WILDLIFE_CLASSES = DETECTION_CLASSES
SPEAKER_FREQUENCIES = {
    "animal_on_road": 15000,
    "person_security_issue": 0,
    "no_incident": 0,
}

_detector_net = None
_model_loaded = False
_model_load_lock = threading.Lock()


class LocalMLUnavailableError(RuntimeError):
    """Raised when local detector inference cannot run."""


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


def _ensure_model_file() -> Path:
    model_path = Path(DETECTOR_MODEL_PATH)
    if model_path.exists():
        return model_path

    if not DETECTOR_MODEL_URL:
        raise LocalMLUnavailableError(f"Detector model missing at {model_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading detector model url=%s path=%s", DETECTOR_MODEL_URL, model_path)
    try:
        urllib.request.urlretrieve(DETECTOR_MODEL_URL, model_path)
    except Exception as exc:
        raise LocalMLUnavailableError(f"Could not download detector model: {exc}") from exc
    return model_path


def load_model():
    """Load the lightweight ONNX detector."""
    global _detector_net, _model_loaded

    if _model_loaded:
        return

    with _model_load_lock:
        if _model_loaded:
            return

        start = time.perf_counter()
        model_path = _ensure_model_file()
        try:
            _detector_net = cv2.dnn.readNetFromONNX(str(model_path))
            _detector_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            _detector_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as exc:
            raise LocalMLUnavailableError(f"Could not load detector model: {exc}") from exc

        _model_loaded = True
        logger.info(
            "Detector model loaded backend=opencv_dnn path=%s elapsed_s=%.2f",
            model_path,
            time.perf_counter() - start,
        )


def _normalize_roi(road_roi: Optional[list[dict]]) -> list[tuple[float, float]]:
    if not road_roi:
        return []
    points = []
    for point in road_roi:
        x = float(point["x"])
        y = float(point["y"])
        points.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return points if len(points) >= 3 else []


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _bbox_road_hit(bbox: list[float], road_polygon: list[tuple[float, float]]) -> bool:
    if not road_polygon:
        return False
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    sample_points = [
        ((x1 + x2) / 2, y2),
        ((x1 + x2) / 2, y1 + height * 0.8),
        (x1 + width * 0.35, y1 + height * 0.8),
        (x1 + width * 0.65, y1 + height * 0.8),
    ]
    return any(_point_in_polygon(point, road_polygon) for point in sample_points)


def _prepare_outputs(outputs) -> np.ndarray:
    output = outputs[0] if isinstance(outputs, tuple) else outputs
    output = np.squeeze(output)
    if output.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)
    if output.shape[0] in {84, 85}:
        output = output.T
    return output


def _detect_objects(image: np.ndarray, confidence_threshold: float) -> list[dict]:
    load_model()
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
        swapRB=True,
        crop=False,
    )
    _detector_net.setInput(blob)
    rows = _prepare_outputs(_detector_net.forward())

    boxes = []
    scores = []
    class_ids = []
    for row in rows:
        if row.shape[0] < 84:
            continue
        if row.shape[0] == 85:
            objectness = float(row[4])
            class_scores = row[5:]
        else:
            objectness = 1.0
            class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id]) * objectness
        if confidence < confidence_threshold or class_id >= len(COCO_CLASSES):
            continue

        cx, cy, w, h = row[:4]
        x1 = max(0, min(width - 1, int((cx - w / 2) * width / DETECTOR_INPUT_SIZE)))
        y1 = max(0, min(height - 1, int((cy - h / 2) * height / DETECTOR_INPUT_SIZE)))
        box_w = max(1, min(width - x1, int(w * width / DETECTOR_INPUT_SIZE)))
        box_h = max(1, min(height - y1, int(h * height / DETECTOR_INPUT_SIZE)))
        boxes.append([x1, y1, box_w, box_h])
        scores.append(confidence)
        class_ids.append(class_id)

    keep = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, DETECTION_NMS_THRESHOLD)
    if len(keep) == 0:
        return []

    detections = []
    for idx in np.array(keep).flatten():
        x, y, box_w, box_h = boxes[idx]
        label = COCO_CLASSES[class_ids[idx]]
        x2 = x + box_w
        y2 = y + box_h
        bbox = [
            round(x / width, 4),
            round(y / height, 4),
            round(x2 / width, 4),
            round(y2 / height, 4),
        ]
        category = "person" if label == "person" else "animal" if label in ANIMAL_CLASSES else "object"
        if category == "object":
            continue
        detections.append(
            {
                "label": label,
                "category": category,
                "confidence": round(float(scores[idx]), 4),
                "bbox": bbox,
            }
        )
    return detections


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_fallen_person(bbox: list[float]) -> bool:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width > height * 1.15


def _update_person_tracks(
    stream_context: dict,
    people_in_road: list[dict],
    now: float,
) -> list[dict]:
    tracks = stream_context.setdefault("person_tracks", [])
    updated_tracks = []
    matched = set()

    for person in people_in_road:
        center = _center(person["bbox"])
        best_index = None
        best_distance = 1.0
        for idx, track in enumerate(tracks):
            if idx in matched:
                continue
            distance = _distance(center, track["positions"][-1][1])
            if distance < best_distance:
                best_index = idx
                best_distance = distance
        if best_index is not None and best_distance < 0.18:
            track = tracks[best_index]
            matched.add(best_index)
        else:
            track = {"first_seen": now, "positions": []}
        track["last_seen"] = now
        track["positions"].append((now, center))
        track["positions"] = [
            item for item in track["positions"] if now - item[0] <= PERSON_ERRATIC_WINDOW_SECONDS
        ]
        person["track_duration_s"] = round(now - track["first_seen"], 2)
        person["erratic"] = _track_is_erratic(track["positions"])
        updated_tracks.append(track)

    stream_context["person_tracks"] = [
        track for track in updated_tracks if now - track.get("last_seen", now) <= PERSON_ERRATIC_WINDOW_SECONDS
    ]
    return people_in_road


def _track_is_erratic(positions: list[tuple[float, tuple[float, float]]]) -> bool:
    if len(positions) < 4:
        return False
    total_distance = sum(
        _distance(positions[i - 1][1], positions[i][1]) for i in range(1, len(positions))
    )
    displacement = _distance(positions[0][1], positions[-1][1])
    return total_distance > 0.25 and total_distance > max(displacement, 0.01) * 2.5


def _people_are_close(people: list[dict]) -> bool:
    for i, first in enumerate(people):
        for second in people[i + 1:]:
            if _distance(_center(first["bbox"]), _center(second["bbox"])) < 0.12:
                return True
    return False


def _apply_road_rules(
    detections: list[dict],
    road_roi: Optional[list[dict]],
    stream_context: Optional[dict],
    now: float,
) -> dict:
    road_polygon = _normalize_roi(road_roi)
    roi_present = bool(road_polygon)
    for detection in detections:
        detection["road_zone_hit"] = _bbox_road_hit(detection["bbox"], road_polygon)

    if not roi_present:
        return {
            "incident_type": None,
            "incident_reason": "missing_road_roi",
            "road_roi_present": False,
            "road_zone_hit": False,
        }

    animal_hits = [
        detection
        for detection in detections
        if detection["category"] == "animal" and detection["road_zone_hit"]
    ]
    if animal_hits:
        best = max(animal_hits, key=lambda item: item["confidence"])
        return {
            "incident_type": "animal_on_road",
            "incident_reason": f"{best['label']}_in_road_roi",
            "road_roi_present": True,
            "road_zone_hit": True,
            "primary_detection": best,
        }

    people_in_road = [
        detection
        for detection in detections
        if detection["category"] == "person" and detection["road_zone_hit"]
    ]
    if not people_in_road:
        return {
            "incident_type": None,
            "incident_reason": "no_relevant_detection_in_road_roi",
            "road_roi_present": True,
            "road_zone_hit": False,
        }

    if stream_context is not None:
        people_in_road = _update_person_tracks(stream_context, people_in_road, now)

    for person in people_in_road:
        if _is_fallen_person(person["bbox"]):
            return _person_incident("fallen_or_lying_in_road", person)
        if person.get("track_duration_s", 0) >= PERSON_LINGER_SECONDS:
            return _person_incident("lingering_in_road", person)
        if person.get("erratic"):
            return _person_incident("erratic_movement_in_road", person)

    if _people_are_close(people_in_road):
        return _person_incident("close_multi_person_interaction_in_road", people_in_road[0])

    return {
        "incident_type": None,
        "incident_reason": "person_in_road_without_abnormal_behavior",
        "road_roi_present": True,
        "road_zone_hit": True,
    }


def _person_incident(reason: str, detection: dict) -> dict:
    return {
        "incident_type": "person_security_issue",
        "incident_reason": reason,
        "road_roi_present": True,
        "road_zone_hit": True,
        "primary_detection": detection,
    }


def predict_frame(
    image: np.ndarray,
    confidence_threshold: Optional[float] = None,
    use_pose_detection: bool = False,
    road_roi: Optional[list[dict]] = None,
    stream_context: Optional[dict] = None,
) -> dict:
    """Detect road-safety incidents in a single frame."""
    del use_pose_detection
    height, width = image.shape[:2]
    threshold = confidence_threshold or DETECTION_CONFIDENCE_THRESHOLD
    start_time = time.perf_counter()
    detections = _detect_objects(image, threshold)
    now = time.monotonic()
    decision = _apply_road_rules(detections, road_roi, stream_context, now)

    primary = decision.get("primary_detection")
    if primary:
        top_label = decision["incident_type"]
        top_confidence = primary["confidence"]
    elif detections:
        best = max(detections, key=lambda item: item["confidence"])
        top_label = f"{best['label']}_detected"
        top_confidence = best["confidence"]
    else:
        top_label = "no_incident"
        top_confidence = 0.0

    predictions = [
        {"species": detection["label"], "confidence": detection["confidence"]}
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True)
    ][:5]
    incident_type = decision.get("incident_type")
    frequency = SPEAKER_FREQUENCIES.get(incident_type or "no_incident", 0)

    return {
        "predicted_species": top_label,
        "confidence": round(top_confidence, 4),
        "alert": incident_type is not None,
        "speaker_frequency_hz": frequency,
        "all_predictions": predictions,
        "frame_width": width,
        "frame_height": height,
        "inference_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "model_backend": "opencv_dnn_yolo_onnx",
        "detections": detections,
        "road_roi_present": decision["road_roi_present"],
        "road_zone_hit": decision["road_zone_hit"],
        "incident_type": incident_type,
        "incident_reason": decision["incident_reason"],
    }


def predict_video(
    video_path: str,
    confidence_threshold: Optional[float] = None,
    sample_fps: Optional[float] = 3.0,
) -> dict:
    """Run frame-level road-safety detection on a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps)) if sample_fps else 1
    frame_count = 0
    processed_count = 0
    frames_results = []
    vote_counts = Counter()
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            result = predict_frame(frame, confidence_threshold)
            label = result["incident_type"] or result["predicted_species"]
            vote_counts[label] += 1
            frames_results.append(
                {
                    "frame_number": frame_count,
                    "timestamp_ms": round((frame_count / video_fps) * 1000, 2),
                    "predicted_species": label,
                    "confidence": result["confidence"],
                }
            )
            processed_count += 1
        frame_count += 1
    cap.release()

    top = vote_counts.most_common(3)
    predicted_species = top[0][0] if top else "no_incident"
    processing_time = time.perf_counter() - start_time
    return {
        "predicted_species": predicted_species,
        "confidence": 1.0 if predicted_species != "no_incident" else 0.0,
        "top3_species": [label for label, _ in top],
        "vote_counts": dict(vote_counts),
        "total_frames": total_frames,
        "frames_processed": processed_count,
        "fps": round(video_fps, 2),
        "processing_fps": round(processed_count / processing_time, 2) if processing_time else 0,
        "frames": frames_results,
    }


def is_model_loaded() -> bool:
    return _model_loaded


def get_model_info() -> dict:
    return {
        "model_name": Path(DETECTOR_MODEL_PATH).name,
        "model_type": "opencv-dnn-yolo-onnx",
        "target_classes": DETECTION_CLASSES,
        "wildlife_classes": sorted(ANIMAL_CLASSES),
        "person_classes": ["person_security_issue"],
        "loaded": is_model_loaded(),
        "device": "cpu" if _model_loaded else "not loaded",
        "accuracy": "rule-based road-safety detector",
        "local_ml_enabled": True,
        "memory_limit_mb": _memory_limit_mb(),
        "detector_model_path": DETECTOR_MODEL_PATH,
        "detection_confidence_threshold": DETECTION_CONFIDENCE_THRESHOLD,
    }
