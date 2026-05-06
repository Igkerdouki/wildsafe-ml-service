#!/usr/bin/env python3
"""
Benchmark YOLO-World on wildlife test clips.

Runs inference on all clips in research/test_data/ and outputs
detection results, FPS metrics, and accuracy comparison.
"""

import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.inference import predict_video, get_model, WILDLIFE_CLASSES


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


def run_benchmark(
    test_data_dir: Path,
    confidence_threshold: float = 0.25,
    sample_fps: float = 5.0
) -> dict:
    """
    Run benchmark on all test clips.

    Returns detailed results including per-clip detections and accuracy metrics.
    """
    print("Loading YOLO-World model...")
    start_load = time.perf_counter()
    get_model()
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    clips = find_test_clips(test_data_dir)
    print(f"Found {len(clips)} test clips across {len(set(c['species'] for c in clips))} species\n")

    results = {
        "model": "yolov8s-worldv2",
        "target_classes": WILDLIFE_CLASSES,
        "confidence_threshold": confidence_threshold,
        "sample_fps": sample_fps,
        "model_load_time_s": round(load_time, 2),
        "clips": [],
        "summary": {}
    }

    total_detections = 0
    species_detections = {}
    fps_values = []

    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] Processing {clip['species']}/{clip['filename']}...")

        try:
            video_result = predict_video(
                clip["path"],
                confidence_threshold=confidence_threshold,
                sample_fps=sample_fps
            )

            # Count detections per class
            detection_counts = {}
            for frame in video_result["frames"]:
                for det in frame["detections"]:
                    label = det["label"]
                    detection_counts[label] = detection_counts.get(label, 0) + 1

            clip_result = {
                "species": clip["species"],
                "filename": clip["filename"],
                "total_frames": video_result["total_frames"],
                "video_fps": video_result["fps"],
                "processing_fps": video_result["processing_fps"],
                "frames_processed": len(video_result["frames"]),
                "detection_counts": detection_counts,
                "total_detections": sum(detection_counts.values())
            }

            # Track metrics
            total_detections += clip_result["total_detections"]
            fps_values.append(video_result["processing_fps"])

            if clip["species"] not in species_detections:
                species_detections[clip["species"]] = {
                    "clips": 0,
                    "detected_as": {},
                    "total_detections": 0
                }

            species_detections[clip["species"]]["clips"] += 1
            species_detections[clip["species"]]["total_detections"] += clip_result["total_detections"]
            for label, count in detection_counts.items():
                sd = species_detections[clip["species"]]["detected_as"]
                sd[label] = sd.get(label, 0) + count

            results["clips"].append(clip_result)
            print(f"    -> {clip_result['total_detections']} detections, {video_result['processing_fps']:.1f} FPS")

        except Exception as e:
            print(f"    -> ERROR: {e}")
            results["clips"].append({
                "species": clip["species"],
                "filename": clip["filename"],
                "error": str(e)
            })

    # Summary statistics
    results["summary"] = {
        "total_clips": len(clips),
        "total_detections": total_detections,
        "avg_processing_fps": round(sum(fps_values) / len(fps_values), 2) if fps_values else 0,
        "species_breakdown": species_detections
    }

    # Compute and add accuracy scores
    scores = compute_accuracy_scores(results)
    results["scores"] = scores

    return results


def compute_accuracy_scores(results: dict) -> dict:
    """
    Compute accuracy scores from benchmark results.

    Returns dict with:
    - species_accuracy: % of detections that match correct species (per species)
    - detection_rate: % of clips with at least one correct detection (per species)
    - overall_accuracy: weighted mean species accuracy
    - overall_detection_rate: mean detection rate across species
    """
    scores = {
        "per_species": {},
        "overall_accuracy": 0.0,
        "overall_detection_rate": 0.0
    }

    # Map species folder names to expected detection labels
    # (handles naming differences like wild_boar -> "wild boar")
    species_to_label = {
        "deer": "deer",
        "raccoon": "raccoon",
        "fox": "fox",
        "coyote": "coyote",
        "opossum": "opossum",
        "skunk": "skunk",
        "bear": "bear",
        "elk": "elk",
        "moose": "moose",
        "goat": "goat",
        "horse": "horse",
        "wild_boar": "wild boar"
    }

    species_breakdown = results["summary"]["species_breakdown"]
    total_correct = 0
    total_detections = 0
    detection_rates = []

    for species, data in species_breakdown.items():
        expected_label = species_to_label.get(species, species.replace("_", " "))
        detected_as = data.get("detected_as", {})

        # Species accuracy: correct detections / total detections
        correct = detected_as.get(expected_label, 0)
        total = data.get("total_detections", 0)
        accuracy = (correct / total * 100) if total > 0 else 0.0

        # Detection rate: check clips for at least one correct detection
        clips_with_correct = 0
        for clip in results["clips"]:
            if clip.get("species") == species:
                clip_counts = clip.get("detection_counts", {})
                if clip_counts.get(expected_label, 0) > 0:
                    clips_with_correct += 1

        clip_count = data.get("clips", 0)
        detection_rate = (clips_with_correct / clip_count * 100) if clip_count > 0 else 0.0

        scores["per_species"][species] = {
            "expected_label": expected_label,
            "correct_detections": correct,
            "total_detections": total,
            "accuracy_pct": round(accuracy, 1),
            "clips_with_correct": clips_with_correct,
            "total_clips": clip_count,
            "detection_rate_pct": round(detection_rate, 1)
        }

        total_correct += correct
        total_detections += total
        detection_rates.append(detection_rate)

    # Overall scores
    scores["overall_accuracy"] = round((total_correct / total_detections * 100) if total_detections > 0 else 0.0, 1)
    scores["overall_detection_rate"] = round(sum(detection_rates) / len(detection_rates) if detection_rates else 0.0, 1)

    return scores


def print_summary(results: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    summary = results["summary"]
    print(f"Model: {results.get('model', 'Unknown')}")
    print(f"Total clips processed: {summary['total_clips']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Average processing FPS: {summary['avg_processing_fps']}")

    # Compute and display accuracy scores
    scores = compute_accuracy_scores(results)

    print("\n" + "=" * 60)
    print("ACCURACY SCORES")
    print("=" * 60)
    print(f"\n>>> OVERALL ACCURACY: {scores['overall_accuracy']}% <<<")
    print(f">>> DETECTION RATE: {scores['overall_detection_rate']}% <<<\n")

    print(f"{'Species':<12} {'Accuracy':>10} {'Det.Rate':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 54)

    for species in sorted(scores["per_species"].keys()):
        s = scores["per_species"][species]
        print(f"{species:<12} {s['accuracy_pct']:>9.1f}% {s['detection_rate_pct']:>9.1f}% {s['correct_detections']:>10} {s['total_detections']:>8}")

    print("\nPer-species detection breakdown:")
    print("-" * 60)
    for species, data in sorted(summary["species_breakdown"].items()):
        print(f"\n{species.upper()} ({data['clips']} clips, {data['total_detections']} total detections)")
        if data["detected_as"]:
            for label, count in sorted(data["detected_as"].items(), key=lambda x: -x[1]):
                print(f"  - {label}: {count}")
        else:
            print("  - No detections")


def main():
    # Paths
    script_dir = Path(__file__).parent
    test_data_dir = script_dir / "test_data"
    output_file = script_dir / "benchmark_results.json"

    if not test_data_dir.exists():
        print(f"Error: Test data directory not found: {test_data_dir}")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(
        test_data_dir,
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
