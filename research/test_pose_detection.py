"""
Test pose detection with various person images.
Downloads sample images and tests the full inference pipeline.
"""

import sys
sys.path.insert(0, '/Users/ioannagkerdouki/wildsafe-ml-service')

import cv2
import numpy as np
import urllib.request
import tempfile
import os

from app.pose_detection import classify_person_state, detect_pose_behavior
from app.inference import predict_frame

# Test images - various person poses (using unsplash source which allows direct downloads)
TEST_IMAGES = {
    "standing_person": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640&q=80",
    "person_sitting": "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?w=640&q=80",
    "person_lying_beach": "https://images.unsplash.com/photo-1520454974749-611b7248ffdb?w=640&q=80",
    "person_planking": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=640&q=80",
}


def download_image(url: str) -> np.ndarray:
    """Download image from URL and return as numpy array."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def test_pose_detection():
    """Test pose detection on various images."""
    print("=" * 60)
    print("POSE DETECTION TEST")
    print("=" * 60)

    for name, url in TEST_IMAGES.items():
        print(f"\nTesting: {name}")
        print("-" * 40)

        image = download_image(url)
        if image is None:
            print("  Failed to download image")
            continue

        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

        # Test pose detection directly
        pose_result = classify_person_state(image)
        print(f"  Pose Detection:")
        print(f"    Detected: {pose_result['detected']}")
        print(f"    State: {pose_result['state']}")
        print(f"    Pose behavior: {pose_result.get('pose_behavior', 'N/A')}")
        print(f"    Body angle: {pose_result.get('body_angle', 'N/A')}°")
        print(f"    Confidence: {pose_result['confidence']:.1%}")
        if pose_result.get('pose_details'):
            details = pose_result['pose_details']
            print(f"    Details:")
            print(f"      Leg angle: {details.get('leg_angle', 'N/A')}°")
            print(f"      Aspect ratio: {details.get('aspect_ratio', 'N/A')}")
            print(f"      Torso/leg ratio: {details.get('torso_leg_ratio', 'N/A')}")

        # Test full inference pipeline
        result = predict_frame(image, confidence_threshold=0.05, use_pose_detection=True)
        print(f"  Full Pipeline:")
        print(f"    Predicted: {result['predicted_species']}")
        print(f"    Confidence: {result['confidence']:.1%}")
        if 'pose_analysis' in result:
            print(f"    Pose state: {result['pose_analysis']['state']}")


def test_angle_calculation():
    """Test the body angle calculation logic."""
    print("\n" + "=" * 60)
    print("BODY ANGLE CLASSIFICATION THRESHOLDS")
    print("=" * 60)
    print("""
    Body Angle | Classification
    -----------|---------------
    0° - 20°   | Standing (upright)
    20° - 40°  | Standing or Sitting (depends on vertical extent)
    40° - 60°  | Hunched/Distress
    > 60°      | Lying Down/Fallen

    Note: Head below hips also triggers 'lying_down' classification.
    """)


if __name__ == "__main__":
    test_angle_calculation()
    test_pose_detection()
