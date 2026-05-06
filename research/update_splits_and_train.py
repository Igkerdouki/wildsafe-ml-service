#!/usr/bin/env python3
"""
Update train/test splits with new clips and retrain classifier.

Usage:
    python research/update_splits_and_train.py
"""

import json
import random
import shutil
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import cv2
from PytorchWildlife.models import detection as pw_detection

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


def update_splits(raw_clips_dir: Path, split_file: Path, test_ratio: float = 0.15):
    """Update train/test splits with all available clips."""
    splits = {"train": {}, "test": {}}

    for species in SPECIES_CLASSES:
        species_dir = raw_clips_dir / species
        if not species_dir.exists():
            continue

        clips = sorted([f.name for f in species_dir.glob("*.mp4")])
        random.shuffle(clips)

        # Reserve ~15% for testing (at least 1)
        num_test = max(1, int(len(clips) * test_ratio))
        test_clips = clips[:num_test]
        train_clips = clips[num_test:]

        splits["test"][species] = test_clips
        splits["train"][species] = train_clips

        print(f"  {species}: {len(train_clips)} train, {len(test_clips)} test")

    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


def extract_crops_from_videos(
    raw_clips_dir: Path,
    train_clips: dict,
    output_dir: Path,
    detector,
    sample_fps: float = 3.0,
    detection_conf: float = 0.25,
    crops_per_clip: int = 50
):
    """Extract animal crops from training video clips."""
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
                continue

            print(f"  [{species}] {clip_name}...", end=" ", flush=True)

            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                print("ERROR: Could not open")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30  # Default fallback
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

                    try:
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
                    except Exception as e:
                        pass  # Skip frame on error

                frame_count += 1

            cap.release()
            print(f"{clip_crops} crops")

    return all_crops, all_labels


def train_classifier(train_loader, val_loader, num_classes, device, epochs=50):
    """Train ResNet18 classifier."""
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
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
    random.seed(42)

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    raw_clips_dir = datasets_dir / "raw_clips"
    video_crops_dir = script_dir / "video_crops"
    models_dir = project_dir / "models"
    split_file = datasets_dir / "splits.json"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 1: Update splits with new clips
    print("\n" + "=" * 60)
    print("STEP 1: Updating train/test splits")
    print("=" * 60)

    splits = update_splits(raw_clips_dir, split_file)

    total_train = sum(len(v) for v in splits["train"].values())
    total_test = sum(len(v) for v in splits["test"].values())
    print(f"\nTotal: {total_train} train clips, {total_test} test clips")

    # Step 2: Load MegaDetector
    print("\n" + "=" * 60)
    print("STEP 2: Loading MegaDetector")
    print("=" * 60)

    detector = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-c")

    # Step 3: Extract crops from training videos
    print("\n" + "=" * 60)
    print("STEP 3: Extracting crops from training videos")
    print("=" * 60)

    if video_crops_dir.exists():
        shutil.rmtree(video_crops_dir)

    video_crops, video_labels = extract_crops_from_videos(
        raw_clips_dir, splits["train"], video_crops_dir, detector,
        sample_fps=3.0, crops_per_clip=50
    )

    print(f"\nTotal video crops: {len(video_crops)}")

    # Count per species
    label_counts = Counter(video_labels)
    print("Crops per species:")
    for species in SPECIES_CLASSES:
        idx = SPECIES_TO_IDX[species]
        print(f"  {species}: {label_counts.get(idx, 0)}")

    # Step 4: Create train/val split from video crops
    print("\n" + "=" * 60)
    print("STEP 4: Creating train/val split from video crops")
    print("=" * 60)

    combined = list(zip(video_crops, video_labels))
    random.shuffle(combined)
    video_crops, video_labels = zip(*combined)
    video_crops, video_labels = list(video_crops), list(video_labels)

    val_split_idx = int(0.15 * len(video_crops))
    val_crops = video_crops[:val_split_idx]
    val_labels = video_labels[:val_split_idx]
    train_crops = video_crops[val_split_idx:]
    train_labels = video_labels[val_split_idx:]

    print(f"Train crops: {len(train_crops)}, Val crops: {len(val_crops)}")

    # Step 5: Create data loaders
    print("\n" + "=" * 60)
    print("STEP 5: Setting up data loaders")
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
    val_dataset = CroppedWildlifeDataset(val_crops, val_labels, val_transform)

    # Weighted sampling for class balance
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Step 6: Train classifier
    print("\n" + "=" * 60)
    print("STEP 6: Training species classifier (video crops only)")
    print("=" * 60)

    model, best_acc = train_classifier(
        train_loader, val_loader,
        num_classes=len(SPECIES_CLASSES),
        device=device,
        epochs=50
    )

    # Step 7: Save model
    print("\n" + "=" * 60)
    print("STEP 7: Saving model")
    print("=" * 60)

    model_path = models_dir / "species_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': SPECIES_CLASSES,
        'best_val_acc': best_acc,
        'model_type': 'resnet18'
    }, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Best validation accuracy: {best_acc:.1f}%")

    print(f"\nNext: python research/evaluate_twostage.py")

    return 0


if __name__ == "__main__":
    exit(main())
