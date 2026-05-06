#!/usr/bin/env python3
"""
Train a species classifier on cropped animal detections.

Two-stage approach:
1. MegaDetector detects animals (95% recall)
2. This classifier identifies the species from the crop

Usage:
    python research/train_classifier.py
"""

import json
import shutil
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import cv2

# Species classes
SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_CLASSES)}


class CroppedWildlifeDataset(Dataset):
    """Dataset of cropped animal images."""

    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def create_crops_from_yolo(datasets_dir: Path, output_dir: Path):
    """Create cropped images from YOLO labels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_images = datasets_dir / "train" / "images"
    train_labels = datasets_dir / "train" / "labels"

    crops = []
    labels = []

    for label_path in sorted(train_labels.glob("*.txt")):
        img_path = train_images / f"{label_path.stem}.jpg"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(label_path) as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                if class_id >= len(SPECIES_CLASSES):
                    continue

                # YOLO format: class x_center y_center width height (normalized)
                x_center, y_center, bw, bh = map(float, parts[1:5])

                # Convert to pixel coordinates
                x1 = int((x_center - bw/2) * w)
                y1 = int((y_center - bh/2) * h)
                x2 = int((x_center + bw/2) * w)
                y2 = int((y_center + bh/2) * h)

                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Skip tiny crops
                if x2 - x1 < 32 or y2 - y1 < 32:
                    continue

                # Crop and save
                crop = img[y1:y2, x1:x2]
                species = SPECIES_CLASSES[class_id]

                species_dir = output_dir / species
                species_dir.mkdir(exist_ok=True)

                crop_name = f"{label_path.stem}_{i}.jpg"
                crop_path = species_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                crops.append(crop_path)
                labels.append(class_id)

    return crops, labels


def train_classifier(train_loader, val_loader, num_classes, device, epochs=20):
    """Train a ResNet18 classifier."""

    # Use ResNet18 pretrained
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    best_model = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict().copy()

        scheduler.step()

    model.load_state_dict(best_model)
    return model, best_acc


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    crops_dir = script_dir / "crops"
    models_dir = project_dir / "models"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 1: Create crops from training data
    print("\n" + "=" * 60)
    print("STEP 1: Creating cropped images from training data")
    print("=" * 60)

    if crops_dir.exists():
        shutil.rmtree(crops_dir)

    crops, labels = create_crops_from_yolo(datasets_dir, crops_dir)
    print(f"Created {len(crops)} crops")

    # Count per species
    label_counts = Counter(labels)
    print("\nCrops per species:")
    for species in SPECIES_CLASSES:
        idx = SPECIES_TO_IDX[species]
        print(f"  {species}: {label_counts.get(idx, 0)}")

    # Step 2: Create train/val split (80/20)
    print("\n" + "=" * 60)
    print("STEP 2: Creating train/val split")
    print("=" * 60)

    # Shuffle
    import random
    random.seed(42)
    combined = list(zip(crops, labels))
    random.shuffle(combined)
    crops, labels = zip(*combined)
    crops, labels = list(crops), list(labels)

    split_idx = int(0.8 * len(crops))
    train_crops, val_crops = crops[:split_idx], crops[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"Train: {len(train_crops)}, Val: {len(val_crops)}")

    # Step 3: Create data loaders
    print("\n" + "=" * 60)
    print("STEP 3: Setting up data loaders")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CroppedWildlifeDataset(train_crops, train_labels, train_transform)
    val_dataset = CroppedWildlifeDataset(val_crops, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Step 4: Train classifier
    print("\n" + "=" * 60)
    print("STEP 4: Training species classifier")
    print("=" * 60)

    model, best_acc = train_classifier(
        train_loader, val_loader,
        num_classes=len(SPECIES_CLASSES),
        device=device,
        epochs=25
    )

    # Step 5: Save model
    print("\n" + "=" * 60)
    print("STEP 5: Saving model")
    print("=" * 60)

    model_path = models_dir / "species_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': SPECIES_CLASSES,
        'best_val_acc': best_acc
    }, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Best validation accuracy: {best_acc:.1f}%")

    # Save class mapping
    info_path = models_dir / "species_classifier_info.json"
    with open(info_path, 'w') as f:
        json.dump({
            'classes': SPECIES_CLASSES,
            'best_val_acc': best_acc,
            'train_samples': len(train_crops),
            'val_samples': len(val_crops)
        }, f, indent=2)

    print(f"\nNext: python research/evaluate_twostage.py")

    return 0


if __name__ == "__main__":
    exit(main())
