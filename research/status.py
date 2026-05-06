#!/usr/bin/env python3
"""
Quick status check for WildSafe project.

Run this first when resuming work to see current state.

Usage:
    python research/status.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    project_dir = Path(__file__).parent.parent
    research_dir = project_dir / "research"

    print()
    print("=" * 60)
    print("  WILDSAFE ML SERVICE - STATUS")
    print("=" * 60)
    print()

    # Latest benchmark
    benchmark_file = research_dir / "benchmark_results.json"
    if benchmark_file.exists():
        with open(benchmark_file) as f:
            benchmark = json.load(f)

        mtime = datetime.fromtimestamp(benchmark_file.stat().st_mtime)
        summary = benchmark.get("summary", {})

        print("## Latest Benchmark")
        print(f"   Run: {mtime.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Model: {benchmark.get('model', 'N/A')}")
        print(f"   Clips: {summary.get('total_clips', 'N/A')}")
        print(f"   Detections: {summary.get('total_detections', 'N/A')}")
        print(f"   Avg FPS: {summary.get('avg_processing_fps', 'N/A')}")
        print()
    else:
        print("## No benchmark results found")
        print("   Run: python research/benchmark.py")
        print()

    # Current model
    inference_file = project_dir / "app" / "inference.py"
    if inference_file.exists():
        content = inference_file.read_text()
        if "yolov8s-worldv2" in content:
            model = "yolov8s-worldv2 (YOLO-World)"
        elif "yolov8" in content.lower():
            model = "YOLOv8 (custom)"
        else:
            model = "Unknown"
        print(f"## Current Model: {model}")
        print()

    # Experiment count
    catalog_file = research_dir / "experiments" / "catalog.md"
    if catalog_file.exists():
        content = catalog_file.read_text()
        exp_count = content.count("## Experiment")
        print(f"## Experiments Run: {exp_count}")
        print(f"   See: research/experiments/catalog.md")
        print()

    # Test data
    test_data_dir = research_dir / "test_data"
    if test_data_dir.exists():
        species = [d for d in test_data_dir.iterdir() if d.is_dir()]
        clips = sum(1 for s in species for f in s.iterdir() if f.suffix in {".mp4", ".webm", ".mkv"})
        print(f"## Test Data: {clips} clips across {len(species)} species")
        print()

    # Git status
    import subprocess
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, cwd=project_dir
        )
        if result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            print(f"## Git: {len(lines)} uncommitted changes")
        else:
            print("## Git: Clean")
    except Exception:
        print("## Git: Unable to check")
    print()

    # Next steps
    print("## Recommended Next Steps")
    print("   1. Review CLAUDE.md for improvement paths")
    print("   2. Check research/experiments/catalog.md for history")
    print("   3. Run: python research/analyze_data.py")
    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
