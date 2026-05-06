#!/usr/bin/env python3
"""
Integrate ENA24 dataset into WildSafe training data.

This script:
1. Extracts ENA24 zip file
2. Maps ENA24 categories to WildSafe species
3. Converts COCO annotations to YOLO format
4. Merges with existing training data
5. Updates dataset.yaml

Usage:
    python research/integrate_ena24.py
"""

import json
import shutil
import zipfile
from pathlib import Path
from collections import Counter


# WildSafe species (must match setup_dataset.py)
WILDSAFE_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]
WILDSAFE_TO_IDX = {name: idx for idx, name in enumerate(WILDSAFE_CLASSES)}

# ENA24 to WildSafe mapping
ENA24_MAPPING = {
    "American Black Bear": "bear",
    "Coyote": "coyote",
    "White_Tailed_Deer": "deer",
    "Red Fox": "fox",
    "Grey Fox": "fox",
    "Horse": "horse",
    "Virginia Opossum": "opossum",
    "Northern Raccoon": "raccoon",
    "Striped Skunk": "skunk",
    # Not mapped (no equivalent or excluded):
    # - Bird, Crow, Turkey, Chicken
    # - Squirrels, Chipmunk, Cottontail, Woodchuck
    # - Human, Vehicle, Dog, Cat, Bobcat
}


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract ENA24 zip file."""
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Already extracted: {extract_dir}")
        return True

    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def load_ena24_annotations(json_path: Path) -> dict:
    """Load ENA24 COCO annotations."""
    with open(json_path) as f:
        data = json.load(f)

    # Build category ID to name mapping
    categories = {c['id']: c['name'] for c in data['categories']}

    # Build image ID to filename mapping
    images = {img['id']: img['file_name'] for img in data['images']}

    # Get image dimensions
    image_dims = {img['id']: (img['width'], img['height']) for img in data['images']}

    return {
        'categories': categories,
        'images': images,
        'image_dims': image_dims,
        'annotations': data['annotations']
    }


def convert_to_yolo(
    ena24_data: dict,
    images_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path
) -> dict:
    """Convert ENA24 annotations to YOLO format."""
    stats = {species: 0 for species in WILDSAFE_CLASSES}
    stats['total_images'] = 0
    stats['total_boxes'] = 0
    stats['skipped_categories'] = Counter()

    # Group annotations by image
    annotations_by_image = {}
    for ann in ena24_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    for img_id, img_filename in ena24_data['images'].items():
        if img_id not in annotations_by_image:
            continue

        annotations = annotations_by_image[img_id]
        width, height = ena24_data['image_dims'][img_id]

        # Filter to mapped categories only
        yolo_lines = []
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = ena24_data['categories'].get(cat_id, "unknown")

            if cat_name not in ENA24_MAPPING:
                stats['skipped_categories'][cat_name] += 1
                continue

            wildsafe_species = ENA24_MAPPING[cat_name]
            class_idx = WILDSAFE_TO_IDX[wildsafe_species]

            # COCO bbox is [x, y, width, height] in pixels
            bbox = ann['bbox']
            x, y, w, h = bbox

            # Convert to YOLO format (center_x, center_y, width, height) normalized
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            # Clamp to valid range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            stats[wildsafe_species] += 1
            stats['total_boxes'] += 1

        if yolo_lines:
            # Copy image
            src_path = images_dir / img_filename
            if not src_path.exists():
                # Try with different extensions
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    alt_path = images_dir / (Path(img_filename).stem + ext)
                    if alt_path.exists():
                        src_path = alt_path
                        break

            if src_path.exists():
                # Create unique filename with ena24 prefix
                new_filename = f"ena24_{Path(img_filename).stem}"
                dst_img = output_images_dir / f"{new_filename}.jpg"
                dst_label = output_labels_dir / f"{new_filename}.txt"

                shutil.copy(src_path, dst_img)

                with open(dst_label, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                stats['total_images'] += 1

    return stats


def update_dataset_yaml(datasets_dir: Path):
    """Update dataset.yaml with new training data."""
    yaml_path = datasets_dir / "dataset.yaml"

    yaml_content = f"""# WildSafe Wildlife Detection Dataset - Combined
# Original clips + ENA24 external data
path: {datasets_dir.absolute()}
train: train/images
val: test/images

# Classes
names:
"""
    for idx, name in enumerate(WILDSAFE_CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Updated {yaml_path}")


def main():
    script_dir = Path(__file__).parent
    datasets_dir = script_dir / "datasets"
    ena24_dir = datasets_dir / "ena24"
    train_dir = datasets_dir / "train"

    # Paths
    zip_path = ena24_dir / "ena24.zip"
    json_path = ena24_dir / "ena24.json"
    images_extract_dir = ena24_dir / "images"

    print("=" * 60)
    print("INTEGRATING ENA24 DATASET")
    print("=" * 60)

    # Check prerequisites
    if not json_path.exists():
        print(f"Error: Annotations not found at {json_path}")
        print("Download from: https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json")
        return 1

    if not zip_path.exists():
        print(f"Error: Images zip not found at {zip_path}")
        print("Download from: https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip")
        return 1

    # Step 1: Extract images
    print("\n--- Step 1: Extracting images ---")
    if not extract_zip(zip_path, images_extract_dir):
        return 1

    # Find actual images directory (might be nested)
    actual_images_dir = images_extract_dir
    if (images_extract_dir / "images").exists():
        actual_images_dir = images_extract_dir / "images"
    elif (images_extract_dir / "ena24").exists():
        actual_images_dir = images_extract_dir / "ena24"
        if (actual_images_dir / "images").exists():
            actual_images_dir = actual_images_dir / "images"

    print(f"Images directory: {actual_images_dir}")

    # Count images
    image_files = list(actual_images_dir.glob("*.*"))
    print(f"Found {len(image_files)} image files")

    # Step 2: Load annotations
    print("\n--- Step 2: Loading annotations ---")
    ena24_data = load_ena24_annotations(json_path)
    print(f"Categories: {len(ena24_data['categories'])}")
    print(f"Images: {len(ena24_data['images'])}")
    print(f"Annotations: {len(ena24_data['annotations'])}")

    # Show mapping
    print("\nENA24 to WildSafe mapping:")
    for ena_cat, ws_cat in ENA24_MAPPING.items():
        print(f"  {ena_cat} -> {ws_cat}")

    # Step 3: Convert and copy to training directory
    print("\n--- Step 3: Converting to YOLO format ---")

    # Count existing training images
    existing_train = list((train_dir / "images").glob("*.jpg"))
    print(f"Existing training images: {len(existing_train)}")

    stats = convert_to_yolo(
        ena24_data,
        actual_images_dir,
        train_dir / "images",
        train_dir / "labels"
    )

    print(f"\nAdded from ENA24:")
    print(f"  Images: {stats['total_images']}")
    print(f"  Bounding boxes: {stats['total_boxes']}")

    print("\nPer-species counts (from ENA24):")
    for species in WILDSAFE_CLASSES:
        if stats[species] > 0:
            print(f"  {species}: {stats[species]}")

    print("\nSkipped categories:")
    for cat, count in stats['skipped_categories'].most_common():
        print(f"  {cat}: {count}")

    # Step 4: Update dataset.yaml
    print("\n--- Step 4: Updating dataset config ---")
    update_dataset_yaml(datasets_dir)

    # Final count
    final_train = list((train_dir / "images").glob("*.jpg"))
    print(f"\n--- SUMMARY ---")
    print(f"Training images before: {len(existing_train)}")
    print(f"Training images after: {len(final_train)}")
    print(f"Images added: {len(final_train) - len(existing_train)}")

    print("\n--- NEXT STEPS ---")
    print("1. Run training: python research/train.py --epochs 50")
    print("2. Evaluate: python research/evaluate.py")

    return 0


if __name__ == "__main__":
    exit(main())
