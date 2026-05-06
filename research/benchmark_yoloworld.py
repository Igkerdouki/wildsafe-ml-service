#!/usr/bin/env python3
"""
Benchmark YOLO-World for wildlife species detection.

YOLO-World is an open-vocabulary detector that can detect any class
you specify at runtime without fine-tuning.
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

# Target wildlife classes for California
WILDLIFE_CLASSES = [
    "deer", "bear", "coyote", "fox", "raccoon", "opossum",
    "skunk", "elk", "moose", "goat", "horse", "wild boar", "boar", "pig"
]

# Map variations to canonical names
CLASS_MAPPING = {
    "boar": "wild_boar",
    "pig": "wild_boar",
    "wild boar": "wild_boar",
}


def evaluate_clip(model, clip_path: Path, ground_truth: str, sample_fps: float = 3.0):
    """Evaluate YOLO-World on a single video clip."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = max(1, int(video_fps / sample_fps))

    vote_counts = defaultdict(int)
    confidence_sums = defaultdict(float)
    total_detections = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Run YOLO-World inference
            results = model(frame, verbose=False)

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id].lower()

                        # Map to canonical name
                        canonical = CLASS_MAPPING.get(class_name, class_name.replace(" ", "_"))

                        if conf > 0.1:  # Low threshold to capture detections
                            vote_counts[canonical] += 1
                            confidence_sums[canonical] += conf
                            total_detections += 1

        frame_count += 1

    cap.release()

    if not vote_counts:
        return {
            "ground_truth": ground_truth,
            "predicted": "none",
            "correct": False,
            "detections": 0,
            "vote_counts": {},
            "avg_confidence": {},
            "top3": [],
            "top3_correct": False
        }

    # Calculate weighted scores (votes * avg_confidence)
    weighted_scores = {}
    for species in vote_counts:
        avg_conf = confidence_sums[species] / vote_counts[species]
        weighted_scores[species] = vote_counts[species] * avg_conf

    # Sort by weighted score
    sorted_species = sorted(weighted_scores.items(), key=lambda x: -x[1])
    predicted = sorted_species[0][0] if sorted_species else "none"
    top3 = [s[0] for s in sorted_species[:3]]

    return {
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": predicted == ground_truth,
        "detections": total_detections,
        "vote_counts": dict(vote_counts),
        "weighted_scores": {k: round(v, 3) for k, v in sorted_species[:5]},
        "top3": top3,
        "top3_correct": ground_truth in top3
    }


def main():
    script_dir = Path(__file__).parent
    raw_clips_dir = script_dir / "datasets" / "raw_clips"
    splits_file = script_dir / "datasets" / "splits.json"

    print("=" * 60)
    print("YOLO-World Wildlife Detection Benchmark")
    print("=" * 60)

    # Load YOLO-World model
    print("\nLoading YOLO-World model...")
    model = YOLO("yolov8s-worldv2.pt")  # Small variant for speed

    # Set custom classes
    print(f"Setting target classes: {WILDLIFE_CLASSES}")
    model.set_classes(WILDLIFE_CLASSES)

    # Load test split from splits.json
    with open(splits_file) as f:
        splits = json.load(f)

    # Find all test clips
    test_clips = []
    for species, clips in splits["test"].items():
        for clip_name in clips:
            clip_path = raw_clips_dir / species / clip_name
            if clip_path.exists():
                test_clips.append((species, clip_path))
            else:
                print(f"Warning: Test clip not found: {clip_path}")

    print(f"\nFound {len(test_clips)} test clips")
    print("-" * 60)

    results = []
    correct = 0
    top3_correct = 0
    per_species = defaultdict(lambda: {"clips": 0, "correct": 0, "top3_correct": 0})

    for species, clip_path in test_clips:
        print(f"\n[{species}] {clip_path.name}...")

        result = evaluate_clip(model, clip_path, species)
        if result is None:
            print("  ERROR: Could not read clip")
            continue

        results.append({
            "species": species,
            "clip": clip_path.name,
            **result
        })

        per_species[species]["clips"] += 1

        if result["correct"]:
            correct += 1
            per_species[species]["correct"] += 1
            print(f"  CORRECT: {result['predicted']}")
        else:
            print(f"  WRONG: predicted={result['predicted']}, truth={species}")

        if result["top3_correct"]:
            top3_correct += 1
            per_species[species]["top3_correct"] += 1

        print(f"  Detections: {result['detections']}")
        print(f"  Top-3: {result['top3']}")
        if result.get('weighted_scores'):
            print(f"  Scores: {result['weighted_scores']}")

    # Summary
    total = len(results)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (YOLO-World)")
    print("=" * 60)
    print(f"\nTest clips: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Ground truth in top-3: {top3_correct}")
    print(f"\nTop-1 accuracy: {100 * correct / total:.1f}%")
    print(f"Top-3 accuracy: {100 * top3_correct / total:.1f}%")

    print("\nPer-species results:")
    print("-" * 60)
    print(f"{'Species':<12} {'Clips':<8} {'Correct':<10} {'Top-3':<10} {'Acc':<10}")
    print("-" * 60)
    for species in sorted(per_species.keys()):
        data = per_species[species]
        acc = 100 * data["correct"] / data["clips"] if data["clips"] > 0 else 0
        print(f"{species:<12} {data['clips']:<8} {data['correct']:<10} {data['top3_correct']:<10} {acc:.0f}%")
    print("-" * 60)

    # Save results
    output = {
        "model": "yolov8s-worldv2",
        "total_clips": total,
        "clips_correct": correct,
        "clips_top3_correct": top3_correct,
        "accuracy": round(100 * correct / total, 1),
        "top3_accuracy": round(100 * top3_correct / total, 1),
        "per_species": dict(per_species),
        "clip_details": results
    }

    output_path = script_dir / "evaluation_yoloworld_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
