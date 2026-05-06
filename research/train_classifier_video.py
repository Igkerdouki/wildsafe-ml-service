#!/usr/bin/env python3
"""
Train species classifier on crops from VIDEO clips only (not camera traps).

This ensures the classifier learns from the same domain as the test clips.

Usage:
    python research/train_classifier_video.py
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
from PytorchWildlife.models import detection as pw_detection

# Species classes
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


def extract_crops_from_videos(
    raw_clips_dir: Path,
    train_clips: dict,
    output_dir: Path,
    detector,
    sample_fps: float = 3.0,
    detection_conf: float = 0.25,
    crops_per_clip: int = 30
):
    """Extract animal crops from training video clips using MegaDetector."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_crops = []
    all_labels = []

    for species, clip_names in train_clips.items():
        species_dir = raw_clips_dir / species
        species_idx = SPECIES_TO_IDX[species]

        crop_out_dir = output_dir / species
        crop_out_dir.mkdir(exist_ok=True)

        for clip_name in clip_names:
            clip_path = species_dir / clip_name
            if not clip_path.exists():
                print(f"  Warning: {clip_path} not found")
                continue

            print(f"  [{species}] {clip_name}...", end=" ")

            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                print("ERROR: Could not open")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(video_fps / sample_fps))

            clip_crops = 0
            frame_count = 0

            while clip_crops < crops_per_clip:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    h, w = frame.shape[:2]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results = detector.single_image_detection(frame_rgb, det_conf_thres=detection_conf)
                    detections = results.get("detections")

                    if detections is not None and hasattr(detections, 'class_id'):
                        for i, class_id in enumerate(detections.class_id):
                            if class_id == 0:  # animal
                                box = detections.xyxy[i]
                                x1, y1, x2, y2 = map(int, box)
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)

                                if x2 - x1 < 64 or y2 - y1 < 64:
                                    continue

                                crop = frame[y1:y2, x1:x2]
                                crop_name = f"{clip_name.replace('.mp4', '')}_{frame_count}_{i}.jpg"
                                crop_path = crop_out_dir / crop_name
                                cv2.imwrite(str(crop_path), crop)

                                all_crops.append(crop_path)
                                all_labels.append(species_idx)
                                clip_crops += 1

                                if clip_crops >= crops_per_clip:
                                    break

                frame_count += 1

            cap.release()
            print(f"{clip_crops} crops")

    return all_crops, all_labels


def train_classifier(train_loader, val_loader, num_classes, device, epochs=30):
    """Train a ResNet18 classifier with better augmentation."""

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
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
    crops_dir = script_dir / "video_crops"
    models_dir = project_dir / "models"

    split_file = datasets_dir / "splits.json"
    raw_clips_dir = datasets_dir / "raw_clips"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(split_file) as f:
        split = json.load(f)

    # Step 1: Load MegaDetector
    print("\n" + "=" * 60)
    print("STEP 1: Loading MegaDetector")
    print("=" * 60)
    detector = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")

    # Step 2: Extract crops from training video clips
    print("\n" + "=" * 60)
    print("STEP 2: Extracting crops from training VIDEO clips")
    print("=" * 60)

    if crops_dir.exists():
        shutil.rmtree(crops_dir)

    crops, labels = extract_crops_from_videos(
        raw_clips_dir, split["train"], crops_dir, detector,
        sample_fps=3.0, crops_per_clip=50
    )

    print(f"\nTotal crops: {len(crops)}")
    label_counts = Counter(labels)
    print("Crops per species:")
    for species in SPECIES_CLASSES:
        idx = SPECIES_TO_IDX[species]
        print(f"  {species}: {label_counts.get(idx, 0)}")

    # Step 3: Create train/val split
    print("\n" + "=" * 60)
    print("STEP 3: Creating train/val split")
    print("=" * 60)

    import random
    random.seed(42)
    combined = list(zip(crops, labels))
    random.shuffle(combined)
    crops, labels = zip(*combined)
    crops, labels = list(crops), list(labels)

    split_idx = int(0.85 * len(crops))
    train_crops, val_crops = crops[:split_idx], crops[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"Train: {len(train_crops)}, Val: {len(val_crops)}")

    # Step 4: Create data loaders with strong augmentation
    print("\n" + "=" * 60)
    print("STEP 4: Setting up data loaders")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
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

    # Step 5: Train classifier
    print("\n" + "=" * 60)
    print("STEP 5: Training species classifier")
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
        'best_val_acc': best_acc
    }, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Best validation accuracy: {best_acc:.1f}%")

    print(f"\nNext: python research/evaluate_twostage.py")

    return 0


if __name__ == "__main__":
    exit(main())
