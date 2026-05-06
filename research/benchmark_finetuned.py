#!/usr/bin/env python3
"""
Benchmark fine-tuned YOLOv8n on wildlife test clips.

This tests the fine-tuned model's species classification accuracy
against the original test clips.

Usage:
    python research/benchmark_finetuned.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# Species classes (must match training order)
SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]


def find_test_clips(test_data_dir: Path) -> list[dict]:
    """Find all video clips organized by species folder."""
    clips = []
    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

    for species_dir in sorted(test_data_dir.iterdir()):
        if not species_dir.is_dir():
            continue

        species = species_dir.name
        for video_file in sorted(species_dir.iterdir()):
            if video_file.suffix.lower() in video_extensions:
                clips.append({
                    "path": str(video_file),
                    "species": species,
                    "filename": video_file.name
                })

    return clips


def process_video(
    model: YOLO,
    video_path: str,
    confidence_threshold: float = 0.25,
    sample_fps: float = 5.0
) -> dict:
    """Process a video with the fine-tuned model and return results."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame sampling
    if sample_fps and sample_fps < video_fps:
        frame_interval = int(video_fps / sample_fps)
    else:
        frame_interval = 1

    # Track detections per class
    detection_counts = {cls: 0 for cls in SPECIES_CLASSES}
    frame_count = 0
    processed_count = 0
    total_detections = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Run detection
            results = model(frame, conf=confidence_threshold, verbose=False)

            # Count detections by class
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        if 0 <= class_id < len(SPECIES_CLASSES):
                            class_name = SPECIES_CLASSES[class_id]
                            detection_counts[class_name] += 1
                            total_detections += 1

            processed_count += 1

        frame_count += 1

    cap.release()
    total_time = time.perf_counter() - start_time
    processing_fps = processed_count / total_time if total_time > 0 else 0

    return {
        "total_frames": total_frames,
        "frames_processed": processed_count,
        "video_fps": round(video_fps, 2),
        "processing_fps": round(processing_fps, 2),
        "detection_counts": detection_counts,
        "total_detections": total_detections
    }


def compute_accuracy(results: dict) -> dict:
    """Compute species accuracy metrics."""
    species_stats = {}

    for clip in results["clips"]:
        if "error" in clip:
            continue

        species = clip["species"]
        if species not in species_stats:
            species_stats[species] = {
                "clips": 0,
                "total_detections": 0,
                "correct_species": 0,
                "wrong_species": 0,
            }

        stats = species_stats[species]
        stats["clips"] += 1
        stats["total_detections"] += clip["total_detections"]

        # Count correct vs incorrect
        for detected_species, count in clip["detection_counts"].items():
            if detected_species == species:
                stats["correct_species"] += count
            else:
                stats["wrong_species"] += count

    # Compute accuracy percentages
    for species, stats in species_stats.items():
        total = stats["correct_species"] + stats["wrong_species"]
        if total > 0:
            stats["accuracy"] = round(100 * stats["correct_species"] / total, 1)
        else:
            stats["accuracy"] = 0.0

        # Detection rate: did we find anything in clips of this species?
        if stats["total_detections"] > 0:
            stats["detection_rate"] = 100.0
        else:
            stats["detection_rate"] = 0.0

    # Overall accuracy
    total_correct = sum(s["correct_species"] for s in species_stats.values())
    total_detections = sum(s["total_detections"] for s in species_stats.values())
    overall_accuracy = round(100 * total_correct / total_detections, 1) if total_detections > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "per_species": species_stats
    }


def run_benchmark(
    test_data_dir: Path,
    model_path: Path,
    confidence_threshold: float = 0.25,
    sample_fps: float = 5.0
) -> dict:
    """Run benchmark on all test clips."""
    print(f"Loading fine-tuned model: {model_path}")
    start_load = time.perf_counter()
    model = YOLO(str(model_path))
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    clips = find_test_clips(test_data_dir)
    print(f"Found {len(clips)} test clips across {len(set(c['species'] for c in clips))} species\n")

    results = {
        "model": str(model_path.name),
        "model_path": str(model_path),
        "confidence_threshold": confidence_threshold,
        "sample_fps": sample_fps,
        "model_load_time_s": round(load_time, 2),
        "clips": [],
        "summary": {}
    }

    fps_values = []

    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] Processing {clip['species']}/{clip['filename']}...", end=" ")

        try:
            video_result = process_video(
                model,
                clip["path"],
                confidence_threshold=confidence_threshold,
                sample_fps=sample_fps
            )

            # Find most common detection
            counts = video_result["detection_counts"]
            most_common = max(counts.items(), key=lambda x: x[1]) if any(counts.values()) else ("none", 0)

            clip_result = {
                "species": clip["species"],
                "filename": clip["filename"],
                "total_frames": video_result["total_frames"],
                "video_fps": video_result["video_fps"],
                "processing_fps": video_result["processing_fps"],
                "frames_processed": video_result["frames_processed"],
                "detection_counts": video_result["detection_counts"],
                "total_detections": video_result["total_detections"],
                "most_common_detection": most_common[0],
                "most_common_count": most_common[1]
            }

            fps_values.append(video_result["processing_fps"])
            results["clips"].append(clip_result)

            # Check if correct
            is_correct = most_common[0] == clip["species"]
            status = "✓" if is_correct else f"✗ ({most_common[0]})"
            print(f"{video_result['total_detections']} det, {video_result['processing_fps']:.1f} FPS {status}")

        except Exception as e:
            print(f"ERROR: {e}")
            results["clips"].append({
                "species": clip["species"],
                "filename": clip["filename"],
                "error": str(e)
            })

    # Compute summary
    accuracy_results = compute_accuracy(results)
    results["summary"] = {
        "total_clips": len(clips),
        "avg_processing_fps": round(sum(fps_values) / len(fps_values), 2) if fps_values else 0,
        "overall_accuracy": accuracy_results["overall_accuracy"],
        "per_species": accuracy_results["per_species"]
    }

    return results


def print_summary(results: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("FINE-TUNED YOLOv8n BENCHMARK SUMMARY")
    print("=" * 70)

    summary = results["summary"]
    print(f"Model: {results['model']}")
    print(f"Total clips processed: {summary['total_clips']}")
    print(f"Average processing FPS: {summary['avg_processing_fps']}")
    print(f"\nOVERALL SPECIES ACCURACY: {summary['overall_accuracy']}%")

    print("\nPer-species breakdown:")
    print("-" * 70)
    print(f"{'Species':<12} {'Clips':<6} {'Detections':<12} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 70)

    for species in SPECIES_CLASSES:
        if species in summary["per_species"]:
            data = summary["per_species"][species]
            print(f"{species:<12} {data['clips']:<6} {data['total_detections']:<12} "
                  f"{data['correct_species']:<10} {data['accuracy']:.1f}%")

    print("-" * 70)


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    test_data_dir = script_dir / "test_data"
    model_path = project_dir / "models" / "wildlife_yolov8n_best.pt"
    output_file = script_dir / "benchmark_finetuned_results.json"

    if not test_data_dir.exists():
        print(f"Error: Test data directory not found: {test_data_dir}")
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("Run fine_tune.py first to create the model.")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(
        test_data_dir,
        model_path,
        confidence_threshold=0.25,
        sample_fps=5.0
    )

    # Print summary
    print_summary(results)

    # Save full results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
