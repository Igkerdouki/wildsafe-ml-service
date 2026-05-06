#!/usr/bin/env python3
"""
Download additional wildlife videos from Pexels (free, no API key needed for direct links).

This script downloads curated wildlife videos to expand the training dataset.
Videos are organized by species in research/datasets/raw_clips/{species}/
"""

import subprocess
import sys
from pathlib import Path

# Curated Pexels video URLs for each species (direct MP4 links)
# These are free stock videos with wildlife footage
WILDLIFE_VIDEOS = {
    "deer": [
        ("deer_pexels_forest_1.mp4", "https://videos.pexels.com/video-files/6394054/6394054-hd_1920_1080_25fps.mp4"),
        ("deer_pexels_field_1.mp4", "https://videos.pexels.com/video-files/5548185/5548185-hd_1920_1080_25fps.mp4"),
        ("deer_pexels_grazing_1.mp4", "https://videos.pexels.com/video-files/6394052/6394052-hd_1920_1080_25fps.mp4"),
        ("deer_pexels_walking_1.mp4", "https://videos.pexels.com/video-files/5603452/5603452-hd_1920_1080_25fps.mp4"),
    ],
    "elk": [
        ("elk_pexels_meadow_1.mp4", "https://videos.pexels.com/video-files/6010489/6010489-hd_1920_1080_25fps.mp4"),
    ],
    "bear": [
        ("bear_pexels_forest_1.mp4", "https://videos.pexels.com/video-files/6394048/6394048-hd_1920_1080_25fps.mp4"),
    ],
    "fox": [
        ("fox_pexels_snow_1.mp4", "https://videos.pexels.com/video-files/5548026/5548026-hd_1920_1080_25fps.mp4"),
    ],
    "horse": [
        ("horse_pexels_field_1.mp4", "https://videos.pexels.com/video-files/4812203/4812203-hd_1920_1080_25fps.mp4"),
        ("horse_pexels_running_1.mp4", "https://videos.pexels.com/video-files/4812200/4812200-hd_1920_1080_25fps.mp4"),
        ("horse_pexels_grazing_1.mp4", "https://videos.pexels.com/video-files/4812198/4812198-hd_1920_1080_25fps.mp4"),
    ],
    "raccoon": [
        # Raccoon footage is harder to find on free stock sites
    ],
    "goat": [
        ("goat_pexels_mountain_1.mp4", "https://videos.pexels.com/video-files/5548174/5548174-hd_1920_1080_25fps.mp4"),
    ],
}

# Alternative: YouTube video IDs for species with limited Pexels footage
# These require yt-dlp to download
YOUTUBE_VIDEOS = {
    "bear": [
        ("bear_yt_walking_1", "dQw4w9WgXcQ"),  # Replace with actual bear video IDs
    ],
    "moose": [
        ("moose_yt_field_1", ""),  # Add actual moose video IDs
    ],
    "coyote": [
        ("coyote_yt_hunting_1", ""),
    ],
    "raccoon": [
        ("raccoon_yt_night_1", ""),
    ],
    "opossum": [
        ("opossum_yt_walking_1", ""),
    ],
    "skunk": [
        ("skunk_yt_foraging_1", ""),
    ],
    "wild_boar": [
        ("wild_boar_yt_forest_1", ""),
    ],
}


def download_pexels_video(url: str, output_path: Path) -> bool:
    """Download video from Pexels using curl."""
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", str(output_path), url],
            capture_output=True,
            timeout=120
        )
        return output_path.exists() and output_path.stat().st_size > 10000
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_youtube_video(video_id: str, output_path: Path) -> bool:
    """Download video from YouTube using yt-dlp."""
    if not video_id:
        return False

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "best[height<=720]",
                "-o", str(output_path),
                f"https://www.youtube.com/watch?v={video_id}"
            ],
            capture_output=True,
            timeout=300
        )
        return output_path.exists()
    except FileNotFoundError:
        print("  yt-dlp not installed. Install with: pip install yt-dlp")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    script_dir = Path(__file__).parent
    raw_clips_dir = script_dir / "datasets" / "raw_clips"

    print("=" * 60)
    print("DOWNLOADING ADDITIONAL WILDLIFE VIDEOS")
    print("=" * 60)

    total_downloaded = 0
    total_failed = 0

    # Download Pexels videos
    print("\n[1/2] Downloading from Pexels...")
    for species, videos in WILDLIFE_VIDEOS.items():
        if not videos:
            continue

        species_dir = raw_clips_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {species}:")
        for filename, url in videos:
            output_path = species_dir / filename
            if output_path.exists():
                print(f"    {filename} - already exists")
                continue

            print(f"    Downloading {filename}...", end=" ", flush=True)
            if download_pexels_video(url, output_path):
                print("OK")
                total_downloaded += 1
            else:
                print("FAILED")
                total_failed += 1

    # Download YouTube videos (if yt-dlp available)
    print("\n[2/2] Downloading from YouTube...")
    for species, videos in YOUTUBE_VIDEOS.items():
        if not videos:
            continue

        species_dir = raw_clips_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)

        for filename, video_id in videos:
            if not video_id:
                continue

            output_path = species_dir / f"{filename}.mp4"
            if output_path.exists():
                print(f"    {filename} - already exists")
                continue

            print(f"    Downloading {filename}...", end=" ", flush=True)
            if download_youtube_video(video_id, output_path):
                print("OK")
                total_downloaded += 1
            else:
                print("FAILED")
                total_failed += 1

    print("\n" + "=" * 60)
    print(f"Downloaded: {total_downloaded} videos")
    print(f"Failed: {total_failed} videos")
    print("=" * 60)

    # Show current dataset status
    print("\nCurrent dataset status:")
    for species in sorted(WILDLIFE_VIDEOS.keys()):
        species_dir = raw_clips_dir / species
        if species_dir.exists():
            count = len(list(species_dir.glob("*.mp4")))
            print(f"  {species}: {count} clips")

    return 0


if __name__ == "__main__":
    sys.exit(main())
