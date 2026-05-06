#!/usr/bin/env python3
"""
Fine-tune YOLOv8n on wildlife species data.

Usage:
    python research/fine_tune.py [--epochs 50] [--batch 16]
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def split_dataset(data_dir: Path, val_ratio: float = 0.2):
    """Split dataset into train/val sets."""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    train_images = data_dir / "train" / "images"
    train_labels = data_dir / "train" / "labels"
    val_images = data_dir / "val" / "images"
    val_labels = data_dir / "val" / "labels"

    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all images
    all_images = sorted(images_dir.glob("*.jpg"))

    # Split by species to ensure balanced val set
    species_images = {}
    for img in all_images:
        species = img.stem.split("_")[0]
        if species not in species_images:
            species_images[species] = []
        species_images[species].append(img)

    train_count = 0
    val_count = 0

    for species, images in species_images.items():
        n_val = max(1, int(len(images) * val_ratio))
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for img in train_imgs:
            label = labels_dir / f"{img.stem}.txt"
            shutil.copy(img, train_images / img.name)
            if label.exists():
                shutil.copy(label, train_labels / label.name)
            train_count += 1

        for img in val_imgs:
            label = labels_dir / f"{img.stem}.txt"
            shutil.copy(img, val_images / img.name)
            if label.exists():
                shutil.copy(label, val_labels / label.name)
            val_count += 1

    print(f"Split dataset: {train_count} train, {val_count} val")
    return train_count, val_count


def update_dataset_yaml(data_dir: Path):
    """Update dataset.yaml with train/val paths."""
    yaml_content = f"""# WildSafe Wildlife Detection Dataset
path: {data_dir.absolute()}
train: train/images
val: val/images

# Classes
names:
  0: bear
  1: coyote
  2: deer
  3: elk
  4: fox
  5: goat
  6: horse
  7: moose
  8: opossum
  9: raccoon
  10: skunk
  11: wild_boar
"""
    yaml_path = data_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on wildlife data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = script_dir / "training_data"
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        print("Error: Training data not found. Run create_training_data.py first.")
        return 1

    # Split dataset
    print("Splitting dataset...")
    split_dataset(data_dir, val_ratio=0.2)

    # Update YAML
    yaml_path = update_dataset_yaml(data_dir)
    print(f"Dataset config: {yaml_path}")

    # Load base model
    print(f"\nLoading base model: {args.model}")
    model = YOLO(args.model)

    # Train
    print(f"\nStarting fine-tuning...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(models_dir),
        name="wildlife_yolov8n",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    # Copy best model to models/
    best_model_src = models_dir / "wildlife_yolov8n" / "weights" / "best.pt"
    best_model_dst = models_dir / "wildlife_yolov8n_best.pt"

    if best_model_src.exists():
        shutil.copy(best_model_src, best_model_dst)
        print(f"\nBest model saved to: {best_model_dst}")

    # Save training info
    info = {
        "timestamp": datetime.now().isoformat(),
        "base_model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "best_model": str(best_model_dst),
        "training_data": str(data_dir),
    }

    info_path = models_dir / "wildlife_yolov8n_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Model: {best_model_dst}")
    print(f"Info: {info_path}")

    return 0


if __name__ == "__main__":
    exit(main())
