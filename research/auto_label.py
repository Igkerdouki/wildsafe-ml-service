#!/usr/bin/env python3
"""
Auto-label images using current model for review.

Usage:
    python auto_label.py <image_dir> [--output-dir labels_review/]

Outputs YOLO format labels for human review.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.inference import predict_frame, get_model, WILDLIFE_CLASSES


def auto_label_directory(
    image_dir: str,
    output_dir: str = None,
    confidence_threshold: float = 0.25
) -> dict:
    """
    Auto-label all images in a directory.

    Args:
        image_dir: Directory containing images
        output_dir: Output directory for labels (default: labels_review/<dirname>)
        confidence_threshold: Minimum confidence for labels

    Returns:
        Summary of labeling results
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    if output_dir is None:
        output_dir = Path(__file__).parent / "labels_review" / image_dir.name
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model
    print("Loading model...")
    get_model()

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    print(f"Found {len(images)} images in {image_dir}")

    results = {
        "total_images": len(images),
        "labeled": 0,
        "no_detections": 0,
        "class_counts": {}
    }

    for i, image_path in enumerate(sorted(images), 1):
        print(f"[{i}/{len(images)}] {image_path.name}...", end=" ")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print("SKIP (could not read)")
            continue

        height, width = image.shape[:2]

        # Run prediction
        prediction = predict_frame(image, confidence_threshold)
        detections = prediction["detections"]

        if not detections:
            print("no detections")
            results["no_detections"] += 1
            continue

        # Convert to YOLO format and save
        label_path = output_dir / f"{image_path.stem}.txt"
        with open(label_path, "w") as f:
            for det in detections:
                label = det["label"]
                bbox = det["bbox"]

                # Get class index
                if label in WILDLIFE_CLASSES:
                    class_idx = WILDLIFE_CLASSES.index(label)
                else:
                    continue

                # Convert to YOLO format (center_x, center_y, width, height) normalized
                x_center = ((bbox["x1"] + bbox["x2"]) / 2) / width
                y_center = ((bbox["y1"] + bbox["y2"]) / 2) / height
                box_width = (bbox["x2"] - bbox["x1"]) / width
                box_height = (bbox["y2"] - bbox["y1"]) / height

                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                # Track counts
                results["class_counts"][label] = results["class_counts"].get(label, 0) + 1

        results["labeled"] += 1
        print(f"{len(detections)} detections")

        # Copy image to review folder
        review_image_path = output_dir / image_path.name
        if not review_image_path.exists():
            cv2.imwrite(str(review_image_path), image)

    # Save summary
    summary_path = output_dir / "labeling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Labels saved to {output_dir}")
    print(f"Labeled: {results['labeled']}, No detections: {results['no_detections']}")
    print(f"Class counts: {results['class_counts']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-label images with current model")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("--output-dir", help="Output directory for labels")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    try:
        auto_label_directory(args.image_dir, args.output_dir, args.confidence)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
