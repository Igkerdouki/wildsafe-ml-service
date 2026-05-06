#!/usr/bin/env python3
"""
Download wildlife videos from YouTube using yt-dlp.

Focus on species with low accuracy in the CLIP evaluation.
"""

import subprocess
import sys
from pathlib import Path

# YouTube video IDs for each species
# Format: (output_name, youtube_video_id)
# Focus on trail camera footage for realistic training data
YOUTUBE_VIDEOS = {
    # Priority: Species with 0% test accuracy need more diverse training data
    "deer": [
        ("deer_trail_relaxing", "3suTUQbML0Y"),  # 1.5 HOURS of Relaxing Trail Camera Videos
        ("deer_maine_wildlife", "h63gmy391CQ"),  # Downeast Maine Wildlife November 2024
        ("deer_best_clips", "pI9KDVRGRCw"),  # 10 Minutes of BEST Trail Camera Clips
        ("deer_michigan_woods", "_49OuAbmIoc"),  # Camera in Michigan Woods
    ],
    "fox": [
        ("fox_night_footage", "cu6nin9vCII"),  # Wildlife Trail Camera Fox Night Footage
        ("fox_family_trail", "6GYOzGuE4sU"),  # Family of cute foxes caught on trail cameras
        ("fox_homestead_spring", "jZfSPFOeEYY"),  # 3 Months trail cameras Wildlife Homestead
        ("fox_backyard_night", "aC_VcJufOjI"),  # Backyard Night Vision Wildlife Camera
    ],
    "horse": [
        ("horse_wild_america", "N0BHMX3IPyY"),  # The Last Of America's Wild Horses Documentary
        ("horse_mustangs_na", "CjDbSzhmF2M"),  # THE WILD MUSTANGS OF NORTH AMERICA
        ("horse_rocky_mountains", "G7lpkPlTe78"),  # North America's Wild Horses
        ("horse_eastern_sierra", "l_Tn5VVCKa4"),  # The Wild Mustangs of the Eastern Sierra
    ],
    "moose": [
        ("moose_alaska_trail", "6e8ax4zOpJg"),  # Alaska Scavengers | Moose Trail Camera
        ("moose_bull_wounded", "nSfLYzxqNKU"),  # Wounded Giant Bull Moose Trail Cameras
        ("moose_maine_wildlife", "D63JO0c4ShY"),  # Maine bears deer moose coyotes wildlife cam
        ("moose_trail_2023", "SeMDlqamQ5Y"),  # Moose and Wildlife Trail Cameras 2023
        ("moose_backcountry_cam", "XYyfxpElQms"),  # Six months at backcountry Moose Cam
        ("moose_minnesota", "JB-1Oo5ygp4"),  # Wild happenings far Northern Minnesota
    ],
    "opossum": [
        ("opossum_curious_close", "bdZqxJ9IjzQ"),  # Curious Opossum Caught on Trail Camera
        ("opossum_walks_by", "H9L1zrYXxCo"),  # Trail Camera: Opossum Walks By
        ("opossum_night_moves", "6Vgq8cKAhXw"),  # Opossum Night Moves Trail Cam
        ("opossum_quirky", "2mor69736L4"),  # Quirky Opossums Trail Camera
        ("opossum_arkansas", "Ry-PxcYgtOU"),  # Rare Opossum Caught On Trail Camera Arkansas
    ],
    "skunk": [
        ("skunk_eating_scraps", "DHAdoA0Syfc"),  # Skunk Eating Vegetable and Fruit Scraps
        ("skunk_homestead_fall", "V4Wmks6oFyE"),  # 3 Months Trail Camera Wildlife Homestead
        ("skunk_wyoming", "cL5ES4VgmIs"),  # Wyoming Wildlife Trail Cam Compilation
        ("skunk_wildlife_oddities", "YTX5wpMMbug"),  # Wildlife Oddities Trail Camera Captures
    ],
    # These species already have good accuracy but adding more for balance
    "raccoon": [
        ("raccoon_maine_rural", "D63JO0c4ShY"),  # Maine wildlife cam (has raccoons too)
    ],
    "coyote": [
        ("coyote_trail_relaxing", "3suTUQbML0Y"),  # 1.5 HOURS Relaxing Trail Camera (has coyotes)
    ],
    "bear": [
        ("bear_maine_trail", "D63JO0c4ShY"),  # Maine wildlife cam (has bears too)
    ],
}


def download_video(video_id: str, output_path: Path, max_duration: int = 120) -> bool:
    """Download a YouTube video."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "best[height<=720]",
                "--max-filesize", "50M",
                "-o", str(output_path),
                "--no-playlist",
                f"https://www.youtube.com/watch?v={video_id}"
            ],
            capture_output=True,
            timeout=300,
            text=True
        )
        return output_path.exists() or any(
            output_path.with_suffix(ext).exists()
            for ext in [".mp4", ".webm", ".mkv"]
        )
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False
    except FileNotFoundError:
        print("yt-dlp not found - install with: pip install yt-dlp")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    script_dir = Path(__file__).parent
    raw_clips_dir = script_dir / "datasets" / "raw_clips"

    print("=" * 60)
    print("DOWNLOADING WILDLIFE VIDEOS FROM YOUTUBE")
    print("=" * 60)

    total_downloaded = 0
    total_failed = 0

    for species, videos in YOUTUBE_VIDEOS.items():
        if not videos:
            continue

        species_dir = raw_clips_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{species}:")
        for name, video_id in videos:
            if not video_id:
                continue

            # Check if already exists
            existing = list(species_dir.glob(f"{name}.*"))
            if existing:
                print(f"  {name} - already exists")
                continue

            output_path = species_dir / f"{name}.mp4"
            print(f"  Downloading {name}...", end=" ", flush=True)

            if download_video(video_id, output_path):
                print("OK")
                total_downloaded += 1
            else:
                print("FAILED")
                total_failed += 1

    print("\n" + "=" * 60)
    print(f"Downloaded: {total_downloaded} videos")
    print(f"Failed: {total_failed} videos")
    print("=" * 60)

    # Show dataset status
    print("\nCurrent dataset status:")
    for species in sorted(YOUTUBE_VIDEOS.keys()):
        species_dir = raw_clips_dir / species
        if species_dir.exists():
            count = len(list(species_dir.glob("*.*")))
            print(f"  {species}: {count} clips")

    return 0


if __name__ == "__main__":
    sys.exit(main())
