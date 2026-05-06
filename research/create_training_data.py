#!/usr/bin/env python3
"""
Create labeled training data from test clips using MegaDetector for bbox detection.

This script:
1. Extracts frames from each video clip
2. Uses MegaDetector to detect animal bounding boxes
3. Labels detections with species from folder name
4. Outputs YOLO format training data

Usage:
    python research/create_training_data.py
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import torch

from PytorchWildlife.models import detection as pw_detection


# Species classes for YOLO (order matters - this is the class index)
SPECIES_CLASSES = [
    "bear",
    "coyote",
    "deer",
    "elk",
    "fox",
    "goat",
    "horse",
    "moose",
    "opossum",
    "raccoon",
    "skunk",
    "wild_boar",
]

# Map folder names to class indices
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_CLASSES)}


def extract_and_label_video(
    video_path: Path,
    species: str,
    model,
    output_dir: Path,
    frames_per_clip: int = 10,
    confidence_threshold: float = 0.3
) -> dict:
    """
    Extract frames from video and create YOLO labels using MegaDetector bboxes.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // frames_per_clip)

    species_idx = SPECIES_TO_IDX.get(species)
    if species_idx is None:
        return {"error": f"Unknown species: {species}"}

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"frames_extracted": 0, "frames_with_detections": 0, "total_boxes": 0}
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0 and saved_count < frames_per_clip:
            height, width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run MegaDetector
            results = model.single_image_detection(frame_rgb, det_conf_thres=confidence_threshold)

            # Check for animal detections
            detections = results.get("detections")
            animal_boxes = []

            if detections is not None and hasattr(detections, 'class_id'):
                for i, class_id in enumerate(detections.class_id):
                    if class_id == 0:  # animal
                        box = detections.xyxy[i]
                        conf = detections.confidence[i]
                        animal_boxes.append((box, conf))

            if animal_boxes:
                # Save image
                img_name = f"{species}_{video_path.stem}_{saved_count:04d}.jpg"
                img_path = images_dir / img_name
                cv2.imwrite(str(img_path), frame)

                # Save YOLO format label
                label_name = f"{species}_{video_path.stem}_{saved_count:04d}.txt"
                label_path = labels_dir / label_name

                with open(label_path, "w") as f:
                    for box, conf in animal_boxes:
                        x1, y1, x2, y2 = box
                        # Convert to YOLO format (center_x, center_y, width, height) normalized
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height

                        f.write(f"{species_idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                        stats["total_boxes"] += 1

                stats["frames_with_detections"] += 1
                saved_count += 1

            stats["frames_extracted"] += 1

        frame_count += 1

    cap.release()
    return stats


def create_dataset_yaml(output_dir: Path):
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# WildSafe Wildlife Detection Dataset
path: {output_dir.absolute()}
train: images
val: images

# Classes
names:
"""
    for idx, name in enumerate(SPECIES_CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path


def main():
    script_dir = Path(__file__).parent
    test_data_dir = script_dir / "test_data"
    output_dir = script_dir / "training_data"

    if output_dir.exists():
        print(f"Removing existing training data at {output_dir}")
        shutil.rmtree(output_dir)

    print("Loading MegaDetector...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")
    print(f"Model loaded on {device}")

    # Find all video clips
    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    clips = []
    for species_dir in sorted(test_data_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        for video_file in sorted(species_dir.iterdir()):
            if video_file.suffix.lower() in video_extensions:
                clips.append((video_file, species))

    print(f"Found {len(clips)} clips across {len(SPECIES_CLASSES)} species\n")

    # Process each clip
    total_stats = {"frames": 0, "detections": 0, "boxes": 0}

    for i, (video_path, species) in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {species}/{video_path.name}...", end=" ")

        stats = extract_and_label_video(
            video_path,
            species,
            model,
            output_dir,
            frames_per_clip=15,  # Extract ~15 frames per clip
            confidence_threshold=0.25
        )

        if "error" in stats:
            print(f"ERROR: {stats['error']}")
        else:
            print(f"{stats['frames_with_detections']} frames, {stats['total_boxes']} boxes")
            total_stats["frames"] += stats["frames_extracted"]
            total_stats["detections"] += stats["frames_with_detections"]
            total_stats["boxes"] += stats["total_boxes"]

    # Create dataset.yaml
    yaml_path = create_dataset_yaml(output_dir)

    # Create classes.txt
    classes_path = output_dir / "classes.txt"
    with open(classes_path, "w") as f:
        for name in SPECIES_CLASSES:
            f.write(f"{name}\n")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING DATA CREATED")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total frames with detections: {total_stats['detections']}")
    print(f"Total bounding boxes: {total_stats['boxes']}")
    print(f"Dataset config: {yaml_path}")
    print(f"Classes: {len(SPECIES_CLASSES)}")

    # Count per species
    print("\nPer-species breakdown:")
    labels_dir = output_dir / "labels"
    for species in SPECIES_CLASSES:
        count = len(list(labels_dir.glob(f"{species}_*.txt")))
        print(f"  {species}: {count} images")

    # Save summary
    summary = {
        "total_frames": total_stats["detections"],
        "total_boxes": total_stats["boxes"],
        "species": SPECIES_CLASSES,
        "per_species": {}
    }
    for species in SPECIES_CLASSES:
        count = len(list(labels_dir.glob(f"{species}_*.txt")))
        summary["per_species"][species] = count

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
