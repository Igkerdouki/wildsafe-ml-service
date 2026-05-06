#!/usr/bin/env python3
"""
Benchmark MegaDetector V6 on wildlife test clips.

This tests the MegaDetector approach for comparison with YOLO-World.
MegaDetector detects: animal, person, vehicle (not species-specific).

Usage:
    python research/benchmark_megadetector.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import torch

# Import PytorchWildlife
from PytorchWildlife.models import detection as pw_detection


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


def process_video_with_megadetector(
    model,
    video_path: str,
    confidence_threshold: float = 0.25,
    sample_fps: float = 5.0
) -> dict:
    """
    Process a video with MegaDetector and return results.
    """
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

    detection_counts = {"animal": 0, "person": 0, "vehicle": 0}
    frame_count = 0
    processed_count = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB for MegaDetector
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detection
            results = model.single_image_detection(frame_rgb, det_conf_thres=confidence_threshold)

            # Count detections by category
            if results and "detections" in results:
                detections = results["detections"]
                if detections is not None and hasattr(detections, 'class_id') and detections.class_id is not None:
                    for class_id in detections.class_id:
                        # MegaDetector classes: 0=animal, 1=person, 2=vehicle
                        if class_id == 0:
                            detection_counts["animal"] += 1
                        elif class_id == 1:
                            detection_counts["person"] += 1
                        elif class_id == 2:
                            detection_counts["vehicle"] += 1

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
        "total_detections": sum(detection_counts.values())
    }


def run_benchmark(
    test_data_dir: Path,
    confidence_threshold: float = 0.25,
    sample_fps: float = 5.0,
    model_version: str = "MDV6-yolov10-c"  # Compact version for speed
) -> dict:
    """
    Run MegaDetector benchmark on all test clips.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading MegaDetector V6 ({model_version})...")
    start_load = time.perf_counter()
    model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version=model_version)
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Classes: {model.CLASS_NAMES}")

    clips = find_test_clips(test_data_dir)
    print(f"Found {len(clips)} test clips across {len(set(c['species'] for c in clips))} species\n")

    results = {
        "model": f"MegaDetectorV6-{model_version}",
        "device": device,
        "classes": model.CLASS_NAMES,
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
        print(f"[{i}/{len(clips)}] Processing {clip['species']}/{clip['filename']}...", end=" ")

        try:
            video_result = process_video_with_megadetector(
                model,
                clip["path"],
                confidence_threshold=confidence_threshold,
                sample_fps=sample_fps
            )

            clip_result = {
                "species": clip["species"],
                "filename": clip["filename"],
                "total_frames": video_result["total_frames"],
                "video_fps": video_result["video_fps"],
                "processing_fps": video_result["processing_fps"],
                "frames_processed": video_result["frames_processed"],
                "detection_counts": video_result["detection_counts"],
                "total_detections": video_result["total_detections"]
            }

            # Track metrics
            total_detections += clip_result["total_detections"]
            fps_values.append(video_result["processing_fps"])

            # Track per-species
            if clip["species"] not in species_detections:
                species_detections[clip["species"]] = {
                    "clips": 0,
                    "animal": 0,
                    "person": 0,
                    "vehicle": 0,
                    "total_detections": 0
                }

            sd = species_detections[clip["species"]]
            sd["clips"] += 1
            sd["animal"] += video_result["detection_counts"]["animal"]
            sd["person"] += video_result["detection_counts"]["person"]
            sd["vehicle"] += video_result["detection_counts"]["vehicle"]
            sd["total_detections"] += clip_result["total_detections"]

            results["clips"].append(clip_result)
            print(f"{clip_result['total_detections']} detections ({video_result['detection_counts']['animal']} animals), {video_result['processing_fps']:.1f} FPS")

        except Exception as e:
            print(f"ERROR: {e}")
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

    return results


def print_summary(results: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("MEGADETECTOR V6 BENCHMARK SUMMARY")
    print("=" * 60)

    summary = results["summary"]
    print(f"Model: {results['model']}")
    print(f"Device: {results['device']}")
    print(f"Total clips processed: {summary['total_clips']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Average processing FPS: {summary['avg_processing_fps']}")

    print("\nPer-species breakdown (MegaDetector detects 'animal', not species):")
    print("-" * 60)

    total_animals = 0
    total_persons = 0

    for species, data in sorted(summary["species_breakdown"].items()):
        print(f"\n{species.upper()} ({data['clips']} clips)")
        print(f"  Animals detected: {data['animal']}")
        print(f"  Persons detected: {data['person']}")
        print(f"  Vehicles detected: {data['vehicle']}")
        total_animals += data["animal"]
        total_persons += data["person"]

    print("\n" + "-" * 60)
    print(f"TOTAL: {total_animals} animals, {total_persons} persons detected")

    # Calculate detection rate (did it find ANY animal in wildlife clips?)
    clips_with_animals = sum(1 for c in results["clips"] if c.get("detection_counts", {}).get("animal", 0) > 0)
    print(f"Clips with animal detections: {clips_with_animals}/{len(results['clips'])} ({clips_with_animals/len(results['clips'])*100:.0f}%)")


def main():
    script_dir = Path(__file__).parent
    test_data_dir = script_dir / "test_data"
    output_file = script_dir / "benchmark_megadetector_results.json"

    if not test_data_dir.exists():
        print(f"Error: Test data directory not found: {test_data_dir}")
        sys.exit(1)

    # Run benchmark with compact model for speed
    results = run_benchmark(
        test_data_dir,
        confidence_threshold=0.25,
        sample_fps=5.0,
        model_version="MDV6-yolov10-c"  # Compact version
    )

    # Print summary
    print_summary(results)

    # Save full results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
