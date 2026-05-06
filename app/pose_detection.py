"""
Pose-based behavior detection using MediaPipe Tasks API.

Detects body orientation and posture to identify:
- Standing/walking (normal)
- Sitting
- Lying down (fallen)
- Hunched/slumped (distress)
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass, field
import urllib.request
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# Pose detection singleton
_pose_detector: Optional[vision.PoseLandmarker] = None


@dataclass
class PoseResult:
    """Result of pose-based behavior detection."""
    person_detected: bool
    behavior: str  # "standing", "sitting", "lying_down", "hunched", "unknown"
    confidence: float
    body_angle: Optional[float] = None
    head_below_hips: bool = False
    details: dict = field(default_factory=dict)


def download_model():
    """Download the pose landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def get_pose_detector() -> vision.PoseLandmarker:
    """Get or create the pose detection model."""
    global _pose_detector
    if _pose_detector is None:
        download_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1
        )
        _pose_detector = vision.PoseLandmarker.create_from_options(options)
    return _pose_detector


def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate angle between two points relative to vertical."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = abs(math.degrees(math.atan2(dx, -dy)))
    return min(angle, 180 - angle)


def analyze_pose(landmarks) -> PoseResult:
    """
    Analyze pose landmarks to determine behavior state.

    MediaPipe pose landmarks:
    - 0: nose
    - 11, 12: left/right shoulder
    - 23, 24: left/right hip
    - 25, 26: left/right knee
    - 27, 28: left/right ankle
    """
    # Extract key points
    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]

    # Calculate midpoints
    shoulder_mid = (
        (left_shoulder.x + right_shoulder.x) / 2,
        (left_shoulder.y + right_shoulder.y) / 2
    )
    hip_mid = (
        (left_hip.x + right_hip.x) / 2,
        (left_hip.y + right_hip.y) / 2
    )
    knee_mid = (
        (left_knee.x + right_knee.x) / 2,
        (left_knee.y + right_knee.y) / 2
    )
    ankle_mid = (
        (left_ankle.x + right_ankle.x) / 2,
        (left_ankle.y + right_ankle.y) / 2
    )

    # Calculate body angle (torso orientation from vertical)
    body_angle = calculate_angle(hip_mid, shoulder_mid)

    # Calculate leg angle (thigh orientation)
    leg_angle = calculate_angle(hip_mid, knee_mid)

    # Check if head is below hips (strong indicator of fallen)
    head_below_hips = nose.y > hip_mid[1]

    # Calculate extents
    y_coords = [nose.y, left_shoulder.y, right_shoulder.y,
                left_hip.y, right_hip.y, left_ankle.y, right_ankle.y]
    x_coords = [nose.x, left_shoulder.x, right_shoulder.x,
                left_hip.x, right_hip.x, left_ankle.x, right_ankle.x]
    vertical_extent = max(y_coords) - min(y_coords)
    horizontal_extent = max(x_coords) - min(x_coords)

    # Aspect ratio: horizontal/vertical - higher means more horizontal (lying)
    aspect_ratio = horizontal_extent / max(vertical_extent, 0.01)

    # Hip-knee relationship (sitting detection)
    # When sitting, knees are often at same level or below hips
    hip_knee_diff = knee_mid[1] - hip_mid[1]  # Positive if knees below hips (in image coords)

    # Torso height vs leg height ratio
    torso_height = abs(shoulder_mid[1] - hip_mid[1])
    leg_height = abs(hip_mid[1] - ankle_mid[1])
    torso_leg_ratio = torso_height / max(leg_height, 0.01)

    details = {
        "body_angle": round(body_angle, 1),
        "leg_angle": round(leg_angle, 1),
        "head_below_hips": head_below_hips,
        "vertical_extent": round(vertical_extent, 3),
        "horizontal_extent": round(horizontal_extent, 3),
        "aspect_ratio": round(aspect_ratio, 2),
        "torso_leg_ratio": round(torso_leg_ratio, 2)
    }

    # Classification logic - improved with multiple features

    # LYING DOWN: High body angle OR head below hips OR very horizontal aspect
    if body_angle > 60 or head_below_hips or aspect_ratio > 2.0:
        behavior = "lying_down"
        # Higher confidence for clearer indicators
        angle_score = min(body_angle / 90, 1.0)
        aspect_score = min(aspect_ratio / 3.0, 1.0) if aspect_ratio > 1.5 else 0
        head_score = 0.3 if head_below_hips else 0
        confidence = min(0.95, 0.6 + 0.2 * angle_score + 0.1 * aspect_score + head_score)

    # HUNCHED/DISTRESS: Moderate forward lean
    elif body_angle > 35:
        behavior = "hunched"
        confidence = 0.65 + (body_angle - 35) / 50 * 0.25

    # SITTING: Small body angle but high leg angle or bent posture
    elif leg_angle > 40 or (vertical_extent < 0.5 and torso_leg_ratio > 0.8):
        behavior = "sitting"
        leg_score = min(leg_angle / 90, 1.0)
        confidence = 0.7 + 0.2 * leg_score

    # STANDING: Upright posture with low angles
    else:
        behavior = "standing"
        # More upright = higher confidence
        confidence = 0.85 + (20 - min(body_angle, 20)) / 20 * 0.1

    return PoseResult(
        person_detected=True,
        behavior=behavior,
        confidence=round(confidence, 3),
        body_angle=round(body_angle, 1),
        head_below_hips=head_below_hips,
        details=details
    )


def detect_pose_behavior(image: np.ndarray) -> PoseResult:
    """
    Detect person pose and classify behavior.

    Args:
        image: BGR image from OpenCV

    Returns:
        PoseResult with behavior classification
    """
    detector = get_pose_detector()

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect pose
    result = detector.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return PoseResult(
            person_detected=False,
            behavior="no_person",
            confidence=0.0,
            details={"error": "No pose landmarks detected"}
        )

    return analyze_pose(result.pose_landmarks[0])


def classify_person_state(image: np.ndarray) -> dict:
    """
    High-level API for person state classification.

    Returns dict with:
    - detected: bool
    - state: "normal", "fallen", "distress", "no_person"
    - confidence: float
    - pose_details: dict with raw pose data
    """
    result = detect_pose_behavior(image)

    if not result.person_detected:
        return {
            "detected": False,
            "state": "no_person",
            "confidence": 0.0,
            "pose_details": result.details
        }

    # Map pose behaviors to alert states
    state_mapping = {
        "standing": "normal",
        "sitting": "normal",
        "lying_down": "fallen",
        "hunched": "distress",
        "unknown": "unknown"
    }

    state = state_mapping.get(result.behavior, "unknown")

    return {
        "detected": True,
        "state": state,
        "pose_behavior": result.behavior,
        "confidence": result.confidence,
        "body_angle": result.body_angle,
        "head_below_hips": result.head_below_hips,
        "pose_details": result.details
    }
