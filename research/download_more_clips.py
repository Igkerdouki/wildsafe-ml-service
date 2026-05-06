#!/usr/bin/env python3
"""
Download additional wildlife video clips from YouTube.

Focuses on species that are failing in classification.

Usage:
    python research/download_more_clips.py
"""

import subprocess
import os
from pathlib import Path
import json
import random

# Search queries for each species - multiple variations for diversity
SEARCH_QUERIES = {
    # Priority species (failing)
    "coyote": [
        "coyote walking wildlife camera",
        "coyote in wild footage",
        "coyote hunting prey nature",
        "urban coyote sighting",
        "coyote howling wild",
    ],
    "goat": [
        "mountain goat climbing rocks",
        "wild goat wildlife footage",
        "mountain goat documentary",
        "ibex wild goat nature",
        "goat on cliff wild",
    ],
    "horse": [
        "wild horse mustang running",
        "wild horses nature documentary",
        "mustang horse wild footage",
        "feral horse herd wild",
        "wild horse grazing",
    ],
    "opossum": [
        "opossum at night wildlife",
        "opossum climbing tree",
        "virginia opossum wild",
        "possum wildlife camera footage",
        "opossum foraging night",
    ],
    "raccoon": [
        "raccoon at night wildlife",
        "raccoon foraging food",
        "wild raccoon footage",
        "raccoon climbing tree",
        "urban raccoon wildlife camera",
    ],
    "skunk": [
        "skunk wildlife camera footage",
        "skunk walking wild",
        "striped skunk nature",
        "skunk foraging night",
        "wild skunk footage",
    ],
    "wild_boar": [
        "wild boar forest footage",
        "wild boar running",
        "wild pig feral hog",
        "wild boar family piglets",
        "wild boar nature documentary",
    ],
    # Secondary species (already working but need more data)
    "bear": [
        "black bear walking forest",
        "grizzly bear wildlife footage",
        "bear fishing salmon",
        "bear in wild nature",
    ],
    "deer": [
        "white tail deer walking",
        "deer in forest wildlife",
        "buck deer antlers",
        "deer grazing field",
    ],
    "elk": [
        "elk bugling wild",
        "elk herd walking",
        "bull elk antlers",
        "elk wildlife footage",
    ],
    "fox": [
        "red fox hunting field",
        "fox in wild nature",
        "fox walking wildlife",
        "red fox wildlife camera",
    ],
    "moose": [
        "moose walking forest",
        "bull moose antlers",
        "moose in wild footage",
        "moose wildlife camera",
    ],
}

# How many new clips to download per species
TARGET_CLIPS = {
    # Priority species - download more
    "coyote": 8,
    "goat": 8,
    "horse": 8,
    "opossum": 8,
    "raccoon": 8,
    "skunk": 10,  # Extra for skunk since detection was failing
    "wild_boar": 8,
    # Secondary species
    "bear": 5,
    "deer": 5,
    "elk": 5,
    "fox": 5,
    "moose": 5,
}


def download_clips_for_species(species: str, queries: list, output_dir: Path, num_clips: int):
    """Download clips for a species using yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    used_videos = set()

    # Load existing video IDs to avoid duplicates
    existing_clips = list(output_dir.glob("*.mp4"))
    print(f"  Existing clips: {len(existing_clips)}")

    for query in queries:
        if downloaded >= num_clips:
            break

        print(f"  Searching: '{query}'...")

        # Search for videos
        search_cmd = [
            "yt-dlp",
            f"ytsearch10:{query}",
            "--get-id",
            "--no-warnings",
        ]

        try:
            result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30)
            video_ids = result.stdout.strip().split('\n')
            video_ids = [v for v in video_ids if v and v not in used_videos]
        except subprocess.TimeoutExpired:
            print(f"    Search timeout, skipping...")
            continue
        except Exception as e:
            print(f"    Search error: {e}")
            continue

        for video_id in video_ids:
            if downloaded >= num_clips:
                break

            if video_id in used_videos:
                continue
            used_videos.add(video_id)

            # Download first 30 seconds of video
            output_file = output_dir / f"{species}_{video_id}.mp4"

            if output_file.exists():
                print(f"    Skipping {video_id} (already exists)")
                continue

            download_cmd = [
                "yt-dlp",
                f"https://www.youtube.com/watch?v={video_id}",
                "-f", "best[height<=720]",
                "--download-sections", "*0-30",
                "-o", str(output_file),
                "--no-warnings",
                "--no-playlist",
                "--socket-timeout", "30",
            ]

            try:
                print(f"    Downloading {video_id}...", end=" ", flush=True)
                result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=120)

                if output_file.exists() and output_file.stat().st_size > 10000:
                    downloaded += 1
                    print(f"OK ({downloaded}/{num_clips})")
                else:
                    print("Failed (no output or too small)")
                    if output_file.exists():
                        output_file.unlink()
            except subprocess.TimeoutExpired:
                print("Timeout")
                if output_file.exists():
                    output_file.unlink()
            except Exception as e:
                print(f"Error: {e}")
                if output_file.exists():
                    output_file.unlink()

    return downloaded


def main():
    script_dir = Path(__file__).parent
    raw_clips_dir = script_dir / "datasets" / "raw_clips"

    print("=" * 60)
    print("DOWNLOADING ADDITIONAL WILDLIFE VIDEO CLIPS")
    print("=" * 60)

    total_downloaded = 0

    # Process priority species first
    priority_species = ["skunk", "coyote", "goat", "horse", "opossum", "raccoon", "wild_boar"]
    secondary_species = ["bear", "deer", "elk", "fox", "moose"]

    for species in priority_species + secondary_species:
        queries = SEARCH_QUERIES.get(species, [])
        target = TARGET_CLIPS.get(species, 5)

        print(f"\n[{species.upper()}] Target: {target} new clips")

        species_dir = raw_clips_dir / species
        downloaded = download_clips_for_species(species, queries, species_dir, target)
        total_downloaded += downloaded

        print(f"  Downloaded: {downloaded} clips")

    print("\n" + "=" * 60)
    print(f"TOTAL DOWNLOADED: {total_downloaded} clips")
    print("=" * 60)

    # Count final totals
    print("\nFinal clip counts per species:")
    for species in sorted(SEARCH_QUERIES.keys()):
        species_dir = raw_clips_dir / species
        count = len(list(species_dir.glob("*.mp4"))) if species_dir.exists() else 0
        print(f"  {species}: {count} clips")

    print("\nNext steps:")
    print("  1. Update splits.json with new clips")
    print("  2. Extract video crops: python research/train_classifier_video.py")
    print("  3. Retrain and evaluate")

    return 0


if __name__ == "__main__":
    exit(main())
