#!/usr/bin/env python3
"""
Evaluate two-stage pipeline with test-time augmentation (TTA) and
confidence-weighted voting for more robust predictions.

Improvements:
- TTA: Multiple augmented predictions per crop, averaged
- Confidence-weighted voting: High-confidence predictions count more
- Top-2 consideration: Consider if ground truth is in top-2 predictions

Usage:
    python research/evaluate_twostage_tta.py
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from PytorchWildlife.models import detection as pw_detection
import numpy as np


SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]


def load_classifier(model_path: Path, device: str):
    """Load the trained species classifier."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_type = checkpoint.get('model_type', 'resnet18')
    if model_type == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, len(SPECIES_CLASSES))
        )
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(SPECIES_CLASSES))

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def get_tta_transforms():
    """Get list of transforms for test-time augmentation."""
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.Resize((224, 224)),
            base_transform
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            base_transform
        ]),
        # Slight zoom
        transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            base_transform
        ]),
        # Slight rotation
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(10, 10)),
            base_transform
        ]),
        # Brightness adjustment
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2),
            base_transform
        ]),
    ]

    return tta_transforms


def classify_crop_tta(model, crop_img, device, tta_transforms):
    """Classify crop with test-time augmentation, return averaged probabilities."""
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    all_probs = []

    with torch.no_grad():
        for transform in tta_transforms:
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    # Average probabilities across augmentations
    avg_probs = np.mean(all_probs, axis=0)[0]
    predicted_idx = np.argmax(avg_probs)
    confidence = avg_probs[predicted_idx]

    return SPECIES_CLASSES[predicted_idx], confidence, avg_probs


def evaluate_clip(
    video_path: Path,
    ground_truth_species: str,
    detector,
    classifier,
    device: str,
    tta_transforms,
    sample_fps: float = 5.0,
    detection_conf: float = 0.2,
    min_crop_size: int = 48
):
    """Evaluate clip with TTA and confidence-weighted voting."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = max(1, int(video_fps / sample_fps))

    # Accumulate confidence-weighted votes
    species_scores = defaultdict(float)
    species_counts = defaultdict(int)
    all_probs_list = []

    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            results = detector.single_image_detection(frame_rgb, det_conf_thres=detection_conf)
            detections = results.get("detections")

            if detections is not None and hasattr(detections, 'class_id'):
                for i, class_id in enumerate(detections.class_id):
                    if class_id == 0:  # animal
                        box = detections.xyxy[i]
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 - x1 < min_crop_size or y2 - y1 < min_crop_size:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        species, conf, probs = classify_crop_tta(
                            classifier, crop, device, tta_transforms
                        )

                        # Confidence-weighted voting
                        # Higher confidence predictions count more
                        weight = conf ** 2  # Square to emphasize high confidence
                        species_scores[species] += weight
                        species_counts[species] += 1
                        all_probs_list.append(probs)
                        total_detections += 1

        frame_count += 1

    cap.release()

    # Determine prediction using weighted scores
    if species_scores:
        predicted_species = max(species_scores.keys(), key=lambda s: species_scores[s])

        # Also compute average probability distribution
        avg_probs = np.mean(all_probs_list, axis=0) if all_probs_list else None

        # Get top-3 predictions by weighted score
        sorted_species = sorted(
            species_scores.keys(),
            key=lambda s: species_scores[s],
            reverse=True
        )[:3]

        # Check if ground truth is in top-3
        top3_contains_gt = ground_truth_species in sorted_species
    else:
        predicted_species = "none"
        avg_probs = None
        sorted_species = []
        top3_contains_gt = False

    is_correct = predicted_species == ground_truth_species

    return {
        "ground_truth": ground_truth_species,
        "predicted": predicted_species,
        "correct": is_correct,
        "detections": total_detections,
        "weighted_scores": {s: round(species_scores[s], 3) for s in sorted_species},
        "vote_counts": dict(species_counts),
        "top3": sorted_species,
        "top3_correct": top3_contains_gt
    }


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    models_dir = project_dir / "models"

    classifier_path = models_dir / "species_classifier.pt"
    split_file = datasets_dir / "splits.json"
    raw_clips_dir = datasets_dir / "raw_clips"

    if not classifier_path.exists():
        print(f"Error: Classifier not found at {classifier_path}")
        return 1

    if not split_file.exists():
        print(f"Error: Split file not found at {split_file}")
        return 1

    with open(split_file) as f:
        splits = json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading MegaDetector...")
    detector = pw_detection.MegaDetectorV6(
        device=device, pretrained=True, version="MDV6-yolov10-c"
    )

    print("Loading species classifier...")
    classifier = load_classifier(classifier_path, device)

    print("Setting up TTA transforms...")
    tta_transforms = get_tta_transforms()

    print("\n" + "=" * 60)
    print("TWO-STAGE EVALUATION WITH TTA + WEIGHTED VOTING")
    print("=" * 60)

    results = {
        "total_clips": 0,
        "clips_correct": 0,
        "clips_top3_correct": 0,
        "per_species": {s: {"clips": 0, "correct": 0, "top3_correct": 0} for s in SPECIES_CLASSES},
        "clip_details": []
    }

    for species, clip_names in splits["test"].items():
        species_dir = raw_clips_dir / species
        for clip_name in clip_names:
            clip_path = species_dir / clip_name
            if not clip_path.exists():
                print(f"  Warning: {clip_path} not found")
                continue

            print(f"\n  [{species}] {clip_name}")

            clip_result = evaluate_clip(
                clip_path,
                species,
                detector,
                classifier,
                device,
                tta_transforms
            )

            if "error" in clip_result:
                print(f"    ERROR: {clip_result['error']}")
                continue

            status = "CORRECT" if clip_result["correct"] else f"WRONG -> {clip_result['predicted']}"
            print(f"    Result: {status}")
            print(f"    Detections: {clip_result['detections']}")
            print(f"    Top-3: {clip_result['top3']}")
            print(f"    Weighted scores: {clip_result['weighted_scores']}")

            results["total_clips"] += 1
            results["per_species"][species]["clips"] += 1

            if clip_result["correct"]:
                results["clips_correct"] += 1
                results["per_species"][species]["correct"] += 1

            if clip_result["top3_correct"]:
                results["clips_top3_correct"] += 1
                results["per_species"][species]["top3_correct"] += 1

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
    print(f"Correctly classified: {results['clips_correct']}")
    print(f"Ground truth in top-3: {results['clips_top3_correct']}")

    if results['total_clips'] > 0:
        acc = 100 * results['clips_correct'] / results['total_clips']
        top3_acc = 100 * results['clips_top3_correct'] / results['total_clips']
        print(f"\nTop-1 accuracy: {acc:.1f}%")
        print(f"Top-3 accuracy: {top3_acc:.1f}%")

    print("\nPer-species results:")
    print("-" * 60)
    print(f"{'Species':<12} {'Clips':<8} {'Correct':<10} {'Top-3':<10} {'Acc':<10}")
    print("-" * 60)

    for species in SPECIES_CLASSES:
        data = results["per_species"][species]
        if data["clips"] > 0:
            acc = 100 * data["correct"] / data["clips"]
            t3 = 100 * data["top3_correct"] / data["clips"]
            print(f"{species:<12} {data['clips']:<8} {data['correct']:<10} {data['top3_correct']:<10} {acc:.0f}%")

    print("-" * 60)

    # Save
    output_file = script_dir / "evaluation_tta_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
