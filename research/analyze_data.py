#!/usr/bin/env python3
"""
Analyze current training/test data distribution.

Usage:
    python analyze_data.py
"""

import json
import sys
from pathlib import Path


def analyze_test_data(test_data_dir: Path) -> dict:
    """Analyze test data clips by species."""
    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    species_data = {}

    for species_dir in sorted(test_data_dir.iterdir()):
        if not species_dir.is_dir():
            continue

        clips = [f for f in species_dir.iterdir() if f.suffix.lower() in video_extensions]
        total_size = sum(f.stat().st_size for f in clips)

        species_data[species_dir.name] = {
            "clips": len(clips),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": [f.name for f in clips]
        }

    return species_data


def analyze_training_data(train_dir: Path) -> dict:
    """Analyze training data by category."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    category_data = {}

    for category_dir in sorted(train_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        images = list(category_dir.rglob("*"))
        images = [f for f in images if f.suffix.lower() in image_extensions]

        # Check for label files
        labels = list(category_dir.rglob("*.txt"))

        category_data[category_dir.name] = {
            "images": len(images),
            "labels": len(labels),
            "labeled_pct": round(len(labels) / len(images) * 100, 1) if images else 0
        }

    return category_data


def analyze_labels_review(review_dir: Path) -> dict:
    """Analyze labels pending review."""
    if not review_dir.exists():
        return {}

    review_data = {}
    for species_dir in sorted(review_dir.iterdir()):
        if not species_dir.is_dir():
            continue

        images = [f for f in species_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        labels = [f for f in species_dir.iterdir() if f.suffix == ".txt" and f.stem != "labeling_summary"]

        review_data[species_dir.name] = {
            "images": len(images),
            "labels": len(labels),
            "pending_review": len(labels)
        }

    return review_data


def main():
    research_dir = Path(__file__).parent
    project_dir = research_dir.parent

    print("=" * 60)
    print("WILDSAFE DATA ANALYSIS")
    print("=" * 60)

    # Test data
    test_data_dir = research_dir / "test_data"
    if test_data_dir.exists():
        print("\n## Test Data (research/test_data/)")
        print("-" * 40)
        test_data = analyze_test_data(test_data_dir)
        total_clips = sum(s["clips"] for s in test_data.values())
        total_size = sum(s["total_size_mb"] for s in test_data.values())

        print(f"Total: {total_clips} clips, {total_size:.1f} MB\n")
        print(f"{'Species':<15} {'Clips':>6} {'Size (MB)':>10}")
        print("-" * 35)
        for species, data in sorted(test_data.items()):
            print(f"{species:<15} {data['clips']:>6} {data['total_size_mb']:>10.1f}")
    else:
        print("\nNo test data found")

    # Training data
    train_dir = project_dir / "data" / "train"
    if train_dir.exists():
        print("\n## Training Data (data/train/)")
        print("-" * 40)
        train_data = analyze_training_data(train_dir)

        if train_data:
            print(f"{'Category':<20} {'Images':>8} {'Labels':>8} {'Labeled %':>10}")
            print("-" * 50)
            for category, data in sorted(train_data.items()):
                print(f"{category:<20} {data['images']:>8} {data['labels']:>8} {data['labeled_pct']:>9.1f}%")
        else:
            print("No training data found (directories are empty)")
    else:
        print("\nNo training data directory found")

    # Labels pending review
    review_dir = research_dir / "labels_review"
    if review_dir.exists():
        print("\n## Labels Pending Review (research/labels_review/)")
        print("-" * 40)
        review_data = analyze_labels_review(review_dir)

        if review_data:
            for species, data in sorted(review_data.items()):
                print(f"{species}: {data['pending_review']} labels to review")
        else:
            print("No labels pending review")

    # Latest benchmark
    benchmark_file = research_dir / "benchmark_results.json"
    if benchmark_file.exists():
        print("\n## Latest Benchmark")
        print("-" * 40)
        with open(benchmark_file) as f:
            benchmark = json.load(f)
        summary = benchmark.get("summary", {})
        print(f"Total detections: {summary.get('total_detections', 'N/A')}")
        print(f"Average FPS: {summary.get('avg_processing_fps', 'N/A')}")
        print(f"Model: {benchmark.get('model', 'N/A')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
