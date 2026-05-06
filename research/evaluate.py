#!/usr/bin/env python3
"""
Evaluate trained model on held-out test set.

This provides true accuracy metrics since test clips were never seen during training.

Usage:
    python research/evaluate.py
"""

import json
from pathlib import Path

import cv2
from ultralytics import YOLO


SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]


def evaluate_on_images(model: YOLO, test_dir: Path, confidence: float = 0.25) -> dict:
    """Evaluate model on test images with ground truth labels."""
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"

    results = {
        "total_images": 0,
        "total_gt_boxes": 0,
        "total_predictions": 0,
        "correct_species": 0,
        "per_species": {s: {"gt": 0, "pred": 0, "correct": 0} for s in SPECIES_CLASSES}
    }

    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Read ground truth
        gt_classes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(SPECIES_CLASSES):
                        gt_classes.append(SPECIES_CLASSES[class_id])
                        results["per_species"][SPECIES_CLASSES[class_id]]["gt"] += 1
                        results["total_gt_boxes"] += 1

        # Run prediction
        img = cv2.imread(str(img_path))
        preds = model(img, conf=confidence, verbose=False)

        pred_classes = []
        for pred in preds:
            if pred.boxes is not None:
                for box in pred.boxes:
                    class_id = int(box.cls[0])
                    if 0 <= class_id < len(SPECIES_CLASSES):
                        pred_class = SPECIES_CLASSES[class_id]
                        pred_classes.append(pred_class)
                        results["per_species"][pred_class]["pred"] += 1
                        results["total_predictions"] += 1

        # Check correctness (simplified: majority class match)
        if gt_classes and pred_classes:
            gt_majority = max(set(gt_classes), key=gt_classes.count)
            pred_majority = max(set(pred_classes), key=pred_classes.count)
            if gt_majority == pred_majority:
                results["correct_species"] += len(pred_classes)
                results["per_species"][gt_majority]["correct"] += len(pred_classes)

        results["total_images"] += 1

    return results


def evaluate_on_clips(model: YOLO, raw_clips_dir: Path, test_clips: dict, confidence: float = 0.25, sample_fps: float = 5.0) -> dict:
    """Evaluate model on held-out test video clips."""
    results = {
        "total_clips": 0,
        "clips_correct": 0,
        "per_species": {s: {"clips": 0, "correct": 0, "detections": 0} for s in SPECIES_CLASSES},
        "clip_details": []
    }

    for species, clip_names in test_clips.items():
        species_dir = raw_clips_dir / species
        for clip_name in clip_names:
            clip_path = species_dir / clip_name
            if not clip_path.exists():
                continue

            # Process video
            cap = cv2.VideoCapture(str(clip_path))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(video_fps / sample_fps))

            detection_counts = {s: 0 for s in SPECIES_CLASSES}
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    preds = model(frame, conf=confidence, verbose=False)
                    for pred in preds:
                        if pred.boxes is not None:
                            for box in pred.boxes:
                                class_id = int(box.cls[0])
                                if 0 <= class_id < len(SPECIES_CLASSES):
                                    detection_counts[SPECIES_CLASSES[class_id]] += 1

                frame_count += 1

            cap.release()

            # Determine predicted species (most detections)
            total_detections = sum(detection_counts.values())
            if total_detections > 0:
                predicted_species = max(detection_counts, key=detection_counts.get)
            else:
                predicted_species = "none"

            is_correct = predicted_species == species

            results["total_clips"] += 1
            results["per_species"][species]["clips"] += 1
            results["per_species"][species]["detections"] += total_detections
            if is_correct:
                results["clips_correct"] += 1
                results["per_species"][species]["correct"] += 1

            results["clip_details"].append({
                "species": species,
                "clip": clip_name,
                "predicted": predicted_species,
                "correct": is_correct,
                "detections": total_detections
            })

    return results


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    model_path = project_dir / "models" / "wildlife_detector_best.pt"

    # Check prerequisites
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train.py first.")
        return 1

    split_file = datasets_dir / "splits.json"
    if not split_file.exists():
        print(f"Error: Split file not found at {split_file}")
        print("Run setup_dataset.py first.")
        return 1

    with open(split_file) as f:
        split = json.load(f)

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Evaluate on test images
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST IMAGES")
    print("=" * 60)

    test_dir = datasets_dir / "test"
    img_results = evaluate_on_images(model, test_dir)

    print(f"\nTest images: {img_results['total_images']}")
    print(f"Ground truth boxes: {img_results['total_gt_boxes']}")
    print(f"Predictions: {img_results['total_predictions']}")

    if img_results['total_predictions'] > 0:
        accuracy = 100 * img_results['correct_species'] / img_results['total_predictions']
        print(f"Species accuracy: {accuracy:.1f}%")

    # Evaluate on test clips
    print("\n" + "=" * 60)
    print("EVALUATION ON HELD-OUT TEST CLIPS")
    print("=" * 60)

    raw_clips_dir = datasets_dir / "raw_clips"
    clip_results = evaluate_on_clips(model, raw_clips_dir, split["test"])

    print(f"\nTest clips: {clip_results['total_clips']}")
    print(f"Clips correctly classified: {clip_results['clips_correct']}")

    if clip_results['total_clips'] > 0:
        clip_accuracy = 100 * clip_results['clips_correct'] / clip_results['total_clips']
        print(f"Clip-level accuracy: {clip_accuracy:.1f}%")

    print("\nPer-species results:")
    print("-" * 50)
    print(f"{'Species':<12} {'Clips':<8} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 50)

    for species in SPECIES_CLASSES:
        data = clip_results["per_species"][species]
        if data["clips"] > 0:
            acc = 100 * data["correct"] / data["clips"]
            print(f"{species:<12} {data['clips']:<8} {data['correct']:<10} {acc:.0f}%")

    print("-" * 50)

    # Detailed clip results
    print("\nDetailed clip results:")
    for detail in clip_results["clip_details"]:
        status = "✓" if detail["correct"] else f"✗ ({detail['predicted']})"
        print(f"  {detail['species']:<12} {detail['clip']:<40} {status}")

    # Save results
    output = {
        "model": str(model_path),
        "image_evaluation": img_results,
        "clip_evaluation": clip_results,
        "summary": {
            "test_clips": clip_results["total_clips"],
            "clip_accuracy": round(100 * clip_results['clips_correct'] / clip_results['total_clips'], 1) if clip_results['total_clips'] > 0 else 0
        }
    }

    output_file = script_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
