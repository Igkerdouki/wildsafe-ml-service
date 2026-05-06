#!/usr/bin/env python3
"""
Train YOLOv8n on the properly split wildlife dataset.

Usage:
    python research/train.py [--epochs 30] [--batch 8]
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on wildlife data")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Check dataset exists
    yaml_path = datasets_dir / "dataset.yaml"
    if not yaml_path.exists():
        print("Error: Dataset not found. Run setup_dataset.py first.")
        return 1

    # Check split info
    split_file = datasets_dir / "splits.json"
    if split_file.exists():
        with open(split_file) as f:
            split = json.load(f)
        print(f"Dataset split:")
        print(f"  Train clips: {split['summary']['train_clips']}")
        print(f"  Test clips: {split['summary']['test_clips']}")

    # Count images
    train_images = list((datasets_dir / "train" / "images").glob("*.jpg"))
    test_images = list((datasets_dir / "test" / "images").glob("*.jpg"))
    print(f"  Train images: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")

    # Load base model
    print(f"\nLoading base model: {args.model}")
    model = YOLO(args.model)

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")

    # Detect device - prefer MPS (Apple Silicon GPU) if available
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"  Device: {device}")

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(models_dir),
        name="wildlife_detector",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    # Copy best model
    best_model_src = models_dir / "wildlife_detector" / "weights" / "best.pt"
    best_model_dst = models_dir / "wildlife_detector_best.pt"

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
        "train_images": len(train_images),
        "test_images": len(test_images),
        "best_model": str(best_model_dst),
        "dataset": str(datasets_dir),
    }

    info_path = models_dir / "wildlife_detector_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Model: {best_model_dst}")
    print(f"Info: {info_path}")
    print(f"\nNext: python research/evaluate.py")

    return 0


if __name__ == "__main__":
    exit(main())
