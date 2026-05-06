#!/usr/bin/env python3
"""
Train species classifier prioritizing VIDEO domain crops.

Strategy:
1. Use video crops for validation (matches test domain)
2. Oversample video crops in training (10x weight)
3. Use ResNet18 (less overfitting) with strong augmentation

Usage:
    python research/train_classifier_video_priority.py
"""

import json
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_CLASSES)}


class CroppedWildlifeDataset(Dataset):
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


def get_ena24_crops(datasets_dir: Path) -> tuple:
    """Get crops from ENA24 camera traps."""
    crops_dir = datasets_dir.parent / "crops"
    crops = []
    labels = []

    for species in SPECIES_CLASSES:
        species_dir = crops_dir / species
        if not species_dir.exists():
            continue
        for img_path in species_dir.glob("*.jpg"):
            crops.append(img_path)
            labels.append(SPECIES_TO_IDX[species])

    return crops, labels


def get_video_crops(datasets_dir: Path) -> tuple:
    """Get crops from video clips."""
    crops_dir = datasets_dir.parent / "video_crops"
    crops = []
    labels = []

    for species in SPECIES_CLASSES:
        species_dir = crops_dir / species
        if not species_dir.exists():
            continue
        for img_path in species_dir.glob("*.jpg"):
            crops.append(img_path)
            labels.append(SPECIES_TO_IDX[species])

    return crops, labels


def train_classifier(train_loader, val_loader, num_classes, device, epochs=40):
    """Train ResNet18 with careful regularization."""

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_model = None

    for epoch in range(epochs):
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
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_model = model.state_dict().copy()

        scheduler.step()

    if best_model:
        model.load_state_dict(best_model)
    return model, best_acc


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    models_dir = project_dir / "models"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 1: Gather crops
    print("\n" + "=" * 60)
    print("STEP 1: Gathering crops from both domains")
    print("=" * 60)

    ena24_crops, ena24_labels = get_ena24_crops(datasets_dir)
    video_crops, video_labels = get_video_crops(datasets_dir)

    print(f"ENA24 crops: {len(ena24_crops)}")
    print(f"Video crops: {len(video_crops)}")

    # Step 2: Split video crops for validation (30% for val)
    print("\n" + "=" * 60)
    print("STEP 2: Creating train/val split (video crops for validation)")
    print("=" * 60)

    random.seed(42)
    video_combined = list(zip(video_crops, video_labels))
    random.shuffle(video_combined)

    val_split_idx = int(0.3 * len(video_combined))
    val_data = video_combined[:val_split_idx]
    video_train_data = video_combined[val_split_idx:]

    val_crops, val_labels = zip(*val_data) if val_data else ([], [])
    video_train_crops, video_train_labels = zip(*video_train_data) if video_train_data else ([], [])

    print(f"Video val crops: {len(val_crops)}")
    print(f"Video train crops: {len(video_train_crops)}")

    # Step 3: Combine training data with video oversampling (5x)
    print("\n" + "=" * 60)
    print("STEP 3: Creating training set with video oversampling (5x)")
    print("=" * 60)

    # Oversample video crops 5x
    OVERSAMPLE_FACTOR = 5
    train_crops = list(ena24_crops)
    train_labels = list(ena24_labels)

    for _ in range(OVERSAMPLE_FACTOR):
        train_crops.extend(video_train_crops)
        train_labels.extend(video_train_labels)

    print(f"ENA24 in training: {len(ena24_crops)}")
    print(f"Video in training (5x oversampled): {len(video_train_crops) * OVERSAMPLE_FACTOR}")
    print(f"Total training crops: {len(train_crops)}")

    # Shuffle training data
    combined = list(zip(train_crops, train_labels))
    random.shuffle(combined)
    train_crops, train_labels = zip(*combined)
    train_crops, train_labels = list(train_crops), list(train_labels)

    # Step 4: Create data loaders
    print("\n" + "=" * 60)
    print("STEP 4: Setting up data loaders")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CroppedWildlifeDataset(train_crops, train_labels, train_transform)
    val_dataset = CroppedWildlifeDataset(list(val_crops), list(val_labels), val_transform)

    # Use weighted sampling for class balance
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Step 5: Train classifier
    print("\n" + "=" * 60)
    print("STEP 5: Training species classifier (ResNet18)")
    print("=" * 60)

    model, best_acc = train_classifier(
        train_loader, val_loader,
        num_classes=len(SPECIES_CLASSES),
        device=device,
        epochs=40
    )

    # Step 6: Save model
    print("\n" + "=" * 60)
    print("STEP 6: Saving model")
    print("=" * 60)

    model_path = models_dir / "species_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': SPECIES_CLASSES,
        'best_val_acc': best_acc,
        'model_type': 'resnet18'
    }, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Best validation accuracy (video domain): {best_acc:.1f}%")

    print(f"\nNext: python research/evaluate_twostage.py")

    return 0


if __name__ == "__main__":
    exit(main())
