#!/usr/bin/env python3
"""
Evaluate two-stage detection pipeline on held-out test clips.

Stage 1: MegaDetector for animal detection
Stage 2: Species classifier on cropped detections

Usage:
    python research/evaluate_twostage.py
"""

import json
from pathlib import Path
from collections import Counter

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from PytorchWildlife.models import detection as pw_detection


SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]


def load_classifier(model_path: Path, device: str):
    """Load the trained species classifier."""
    checkpoint = torch.load(model_path, map_location=device)

    # Check model type
    model_type = checkpoint.get('model_type', 'resnet18')
    if model_type == 'resnet50':
        model = models.resnet50(weights=None)
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, len(SPECIES_CLASSES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return model, transform


def classify_crop(model, transform, crop_img, device):
    """Classify a cropped animal image."""
    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    # Transform and predict
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    return SPECIES_CLASSES[predicted.item()], confidence.item()


def evaluate_clip(
    video_path: Path,
    ground_truth_species: str,
    detector,
    classifier,
    classifier_transform,
    device: str,
    sample_fps: float = 5.0,
    detection_conf: float = 0.25,
    classification_conf: float = 0.3
):
    """Evaluate a single video clip with two-stage pipeline."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / sample_fps))

    species_votes = Counter()
    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Stage 1: Detect animals with MegaDetector
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.single_image_detection(frame_rgb, det_conf_thres=detection_conf)

            detections = results.get("detections")
            if detections is not None and hasattr(detections, 'class_id'):
                h, w = frame.shape[:2]

                for i, class_id in enumerate(detections.class_id):
                    if class_id == 0:  # animal class
                        # Get bounding box
                        box = detections.xyxy[i]
                        x1, y1, x2, y2 = map(int, box)

                        # Clamp to image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # Skip tiny detections
                        if x2 - x1 < 32 or y2 - y1 < 32:
                            continue

                        # Stage 2: Classify the crop
                        crop = frame[y1:y2, x1:x2]
                        species, conf = classify_crop(
                            classifier, classifier_transform, crop, device
                        )

                        if conf >= classification_conf:
                            species_votes[species] += 1
                            total_detections += 1

        frame_count += 1

    cap.release()

    # Determine predicted species (majority vote)
    if species_votes:
        predicted_species = species_votes.most_common(1)[0][0]
    else:
        predicted_species = "none"

    is_correct = predicted_species == ground_truth_species

    return {
        "ground_truth": ground_truth_species,
        "predicted": predicted_species,
        "correct": is_correct,
        "detections": total_detections,
        "votes": dict(species_votes)
    }


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    models_dir = project_dir / "models"

    # Paths
    classifier_path = models_dir / "species_classifier.pt"
    split_file = datasets_dir / "splits.json"
    raw_clips_dir = datasets_dir / "raw_clips"

    # Check prerequisites
    if not classifier_path.exists():
        print(f"Error: Classifier not found at {classifier_path}")
        print("Run train_classifier.py first.")
        return 1

    if not split_file.exists():
        print(f"Error: Split file not found at {split_file}")
        return 1

    with open(split_file) as f:
        split = json.load(f)

    # Load models
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading MegaDetector...")
    detector = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")

    print("Loading species classifier...")
    classifier, classifier_transform = load_classifier(classifier_path, device)

    # Evaluate on test clips
    print("\n" + "=" * 60)
    print("TWO-STAGE EVALUATION ON HELD-OUT TEST CLIPS")
    print("=" * 60)

    results = {
        "total_clips": 0,
        "clips_correct": 0,
        "per_species": {s: {"clips": 0, "correct": 0} for s in SPECIES_CLASSES},
        "clip_details": []
    }

    for species, clip_names in split["test"].items():
        species_dir = raw_clips_dir / species
        for clip_name in clip_names:
            clip_path = species_dir / clip_name
            if not clip_path.exists():
                print(f"  Warning: {clip_path} not found")
                continue

            print(f"  [{species}] {clip_name}...", end=" ")

            clip_result = evaluate_clip(
                clip_path,
                species,
                detector,
                classifier,
                classifier_transform,
                device
            )

            if "error" in clip_result:
                print(f"ERROR: {clip_result['error']}")
                continue

            status = "✓" if clip_result["correct"] else f"✗ ({clip_result['predicted']})"
            print(status)

            results["total_clips"] += 1
            results["per_species"][species]["clips"] += 1

            if clip_result["correct"]:
                results["clips_correct"] += 1
                results["per_species"][species]["correct"] += 1

            results["clip_details"].append({
                "species": species,
                "clip": clip_name,
                **clip_result
            })

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nTest clips: {results['total_clips']}")
    print(f"Clips correctly classified: {results['clips_correct']}")

    if results['total_clips'] > 0:
        accuracy = 100 * results['clips_correct'] / results['total_clips']
        print(f"Clip-level accuracy: {accuracy:.1f}%")

    print("\nPer-species results:")
    print("-" * 50)
    print(f"{'Species':<12} {'Clips':<8} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 50)

    for species in SPECIES_CLASSES:
        data = results["per_species"][species]
        if data["clips"] > 0:
            acc = 100 * data["correct"] / data["clips"]
            print(f"{species:<12} {data['clips']:<8} {data['correct']:<10} {acc:.0f}%")

    print("-" * 50)

    # Detailed results
    print("\nDetailed clip results:")
    for detail in results["clip_details"]:
        status = "✓" if detail["correct"] else f"✗ ({detail['predicted']})"
        print(f"  {detail['species']:<12} {detail['clip']:<40} {status}")

    # Save results
    output_file = script_dir / "evaluation_twostage_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
