#!/usr/bin/env python3
"""
Extract frames from video clips for labeling.

Usage:
    python extract_frames.py <video_path> [--fps 2] [--output-dir frames/]
"""

import argparse
import sys
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str = None,
    fps: float = 2.0
) -> list[str]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to video file
        output_dir: Output directory (default: same as video)
        fps: Frames per second to extract

    Returns:
        List of extracted frame paths
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.parent / "frames" / video_path.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if fps < video_fps else 1

    extracted = []
    frame_count = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_path = output_dir / f"{video_path.stem}_{save_count:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted.append(str(output_path))
            save_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {save_count} frames to {output_dir}")
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract")
    parser.add_argument("--output-dir", help="Output directory")

    args = parser.parse_args()

    try:
        frames = extract_frames(args.video_path, args.output_dir, args.fps)
        print(f"Done! {len(frames)} frames extracted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
