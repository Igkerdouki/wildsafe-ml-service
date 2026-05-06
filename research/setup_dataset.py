#!/usr/bin/env python3
"""
Setup properly organized dataset with train/test split at clip level.

This script:
1. Reorganizes folder structure
2. Creates train/test split (1 clip per species held out for testing)
3. Extracts frames using MegaDetector for bounding boxes
4. Outputs YOLO format data for both train and test sets

Usage:
    python research/setup_dataset.py
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import torch
from PytorchWildlife.models import detection as pw_detection


# Species classes (order matters - this is the class index)
SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_CLASSES)}

# Configuration
FRAMES_PER_CLIP = 15
CONFIDENCE_THRESHOLD = 0.25
RANDOM_SEED = 42


def find_clips(raw_clips_dir: Path) -> dict[str, list[Path]]:
    """Find all video clips organized by species."""
    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    clips_by_species = {}

    for species_dir in sorted(raw_clips_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        if species not in SPECIES_TO_IDX:
            print(f"Warning: Unknown species folder '{species}', skipping")
            continue

        clips = []
        for video_file in sorted(species_dir.iterdir()):
            if video_file.suffix.lower() in video_extensions:
                clips.append(video_file)

        if clips:
            clips_by_species[species] = clips

    return clips_by_species


def create_split(clips_by_species: dict[str, list[Path]], seed: int = 42) -> dict:
    """Create train/test split - hold out 1 clip per species for testing."""
    random.seed(seed)

    split = {
        "seed": seed,
        "train": {},
        "test": {},
        "summary": {
            "train_clips": 0,
            "test_clips": 0,
            "species": len(clips_by_species)
        }
    }

    for species, clips in clips_by_species.items():
        # Shuffle clips
        clips_shuffled = clips.copy()
        random.shuffle(clips_shuffled)

        # Hold out 1 for testing, rest for training
        test_clip = clips_shuffled[0]
        train_clips = clips_shuffled[1:]

        split["test"][species] = [test_clip.name]
        split["train"][species] = [c.name for c in train_clips]

        split["summary"]["test_clips"] += 1
        split["summary"]["train_clips"] += len(train_clips)

    return split


def extract_frames_with_labels(
    video_path: Path,
    species: str,
    model,
    output_images_dir: Path,
    output_labels_dir: Path,
    frames_per_clip: int = 15,
    confidence_threshold: float = 0.25
) -> dict:
    """Extract frames and create YOLO labels using MegaDetector."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // frames_per_clip)
    species_idx = SPECIES_TO_IDX[species]

    stats = {"frames": 0, "detections": 0, "boxes": 0}
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

            # Get animal detections
            detections = results.get("detections")
            animal_boxes = []

            if detections is not None and hasattr(detections, 'class_id'):
                for i, class_id in enumerate(detections.class_id):
                    if class_id == 0:  # animal
                        box = detections.xyxy[i]
                        animal_boxes.append(box)

            if animal_boxes:
                # Save image
                img_name = f"{species}_{video_path.stem}_{saved_count:04d}.jpg"
                cv2.imwrite(str(output_images_dir / img_name), frame)

                # Save YOLO label
                label_name = f"{species}_{video_path.stem}_{saved_count:04d}.txt"
                with open(output_labels_dir / label_name, "w") as f:
                    for box in animal_boxes:
                        x1, y1, x2, y2 = box
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        f.write(f"{species_idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                        stats["boxes"] += 1

                stats["detections"] += 1
                saved_count += 1

            stats["frames"] += 1
        frame_count += 1

    cap.release()
    return stats


def create_dataset_yaml(datasets_dir: Path, split_name: str) -> Path:
    """Create YOLO dataset.yaml file."""
    yaml_content = f"""# WildSafe Wildlife Detection Dataset - {split_name}
path: {datasets_dir.absolute()}
train: train/images
val: test/images

# Classes
names:
"""
    for idx, name in enumerate(SPECIES_CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = datasets_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def main():
    script_dir = Path(__file__).parent

    # Paths
    old_test_data = script_dir / "test_data"
    datasets_dir = script_dir / "datasets"
    raw_clips_dir = datasets_dir / "raw_clips"
    train_dir = datasets_dir / "train"
    test_dir = datasets_dir / "test"

    # Step 1: Reorganize folders
    print("=" * 60)
    print("STEP 1: Reorganizing folder structure")
    print("=" * 60)

    if old_test_data.exists() and not raw_clips_dir.exists():
        print(f"Moving {old_test_data} → {raw_clips_dir}")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_test_data), str(raw_clips_dir))
    elif raw_clips_dir.exists():
        print(f"Raw clips already at {raw_clips_dir}")
    else:
        print(f"ERROR: No clips found at {old_test_data} or {raw_clips_dir}")
        return 1

    # Step 2: Find clips and create split
    print("\n" + "=" * 60)
    print("STEP 2: Creating train/test split")
    print("=" * 60)

    clips_by_species = find_clips(raw_clips_dir)
    print(f"Found {sum(len(c) for c in clips_by_species.values())} clips across {len(clips_by_species)} species")

    split = create_split(clips_by_species, seed=RANDOM_SEED)

    # Save split
    split_file = datasets_dir / "splits.json"
    with open(split_file, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Split saved to {split_file}")
    print(f"  Train: {split['summary']['train_clips']} clips")
    print(f"  Test: {split['summary']['test_clips']} clips")

    # Step 3: Clear old extracted data
    print("\n" + "=" * 60)
    print("STEP 3: Preparing output directories")
    print("=" * 60)

    for d in [train_dir, test_dir]:
        if d.exists():
            print(f"Removing existing {d.name}/ directory")
            shutil.rmtree(d)

    for subdir in ["images", "labels"]:
        (train_dir / subdir).mkdir(parents=True)
        (test_dir / subdir).mkdir(parents=True)

    # Step 4: Load MegaDetector
    print("\n" + "=" * 60)
    print("STEP 4: Loading MegaDetector")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")
    print("Model loaded")

    # Step 5: Extract frames for train and test sets
    print("\n" + "=" * 60)
    print("STEP 5: Extracting frames and labels")
    print("=" * 60)

    total_stats = {"train": {"frames": 0, "boxes": 0}, "test": {"frames": 0, "boxes": 0}}

    for split_name, split_data in [("train", split["train"]), ("test", split["test"])]:
        output_dir = train_dir if split_name == "train" else test_dir
        print(f"\n--- {split_name.upper()} SET ---")

        for species, clip_names in split_data.items():
            species_dir = raw_clips_dir / species
            for clip_name in clip_names:
                clip_path = species_dir / clip_name
                if not clip_path.exists():
                    print(f"  Warning: {clip_path} not found")
                    continue

                print(f"  [{species}] {clip_name}...", end=" ")
                stats = extract_frames_with_labels(
                    clip_path,
                    species,
                    model,
                    output_dir / "images",
                    output_dir / "labels",
                    frames_per_clip=FRAMES_PER_CLIP,
                    confidence_threshold=CONFIDENCE_THRESHOLD
                )

                if "error" in stats:
                    print(f"ERROR: {stats['error']}")
                else:
                    print(f"{stats['detections']} frames, {stats['boxes']} boxes")
                    total_stats[split_name]["frames"] += stats["detections"]
                    total_stats[split_name]["boxes"] += stats["boxes"]

    # Step 6: Create dataset.yaml
    print("\n" + "=" * 60)
    print("STEP 6: Creating dataset configuration")
    print("=" * 60)

    yaml_path = create_dataset_yaml(datasets_dir, "train/test split")
    print(f"Dataset config: {yaml_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SETUP COMPLETE")
    print("=" * 60)
    print(f"\nStructure:")
    print(f"  {datasets_dir}/")
    print(f"  ├── raw_clips/        ({sum(len(c) for c in clips_by_species.values())} videos)")
    print(f"  ├── train/")
    print(f"  │   ├── images/       ({total_stats['train']['frames']} images)")
    print(f"  │   └── labels/       ({total_stats['train']['boxes']} boxes)")
    print(f"  ├── test/")
    print(f"  │   ├── images/       ({total_stats['test']['frames']} images)")
    print(f"  │   └── labels/       ({total_stats['test']['boxes']} boxes)")
    print(f"  ├── splits.json")
    print(f"  └── dataset.yaml")

    print(f"\nNext steps:")
    print(f"  1. Train: python research/train.py")
    print(f"  2. Test:  python research/evaluate.py")

    return 0


if __name__ == "__main__":
    exit(main())
