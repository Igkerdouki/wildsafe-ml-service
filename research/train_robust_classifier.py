#!/usr/bin/env python3
"""
Train a robust species classifier with focus on generalization.

Key improvements:
- Heavy augmentation to prevent overfitting
- MixUp/CutMix for better regularization
- Focal loss for hard example mining
- Larger model (ResNet50) with dropout
- Cross-validation style training

Usage:
    python research/train_robust_classifier.py
"""

import json
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FocalLoss(nn.Module):
    """Focal loss for hard example mining."""
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


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


def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation."""
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss computation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def extract_crops_from_videos(
    raw_clips_dir: Path,
    train_clips: dict,
    output_dir: Path,
    detector,
    sample_fps: float = 3.0,
    detection_conf: float = 0.2,  # Lower threshold to get more crops
    crops_per_clip: int = 80      # More crops per clip
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
                print("ERROR")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30
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
                        results = detector.single_image_detection(
                            frame_rgb, det_conf_thres=detection_conf
                        )
                        detections = results.get("detections")

                        if detections is not None and hasattr(detections, 'class_id'):
                            for i, class_id in enumerate(detections.class_id):
                                if class_id == 0:  # animal
                                    box = detections.xyxy[i]
                                    x1, y1, x2, y2 = map(int, box)
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w, x2), min(h, y2)

                                    # Skip small detections
                                    if x2 - x1 < 48 or y2 - y1 < 48:
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
                    except Exception:
                        pass

                frame_count += 1

            cap.release()
            print(f"{clip_crops} crops")

    return all_crops, all_labels


def train_classifier(
    train_loader,
    val_loader,
    num_classes,
    device,
    epochs=60,
    use_mixup=True
):
    """Train ResNet50 classifier with focal loss and mixup."""
    # Use ResNet50 for more capacity
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Add dropout before final layer
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)

    criterion = FocalLoss(alpha=1, gamma=2, label_smoothing=0.1)

    # Lower LR with warm-up
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)

    # Warm-up + cosine annealing
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor(
            (epoch - warmup_epochs) / (epochs - warmup_epochs) * 3.14159
        )).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    best_model = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply MixUp with 50% probability
            if use_mixup and random.random() < 0.5:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_loss += loss.item()

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
        avg_loss = train_loss / len(train_loader)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Train: {train_acc:.1f}%, Val: {val_acc:.1f}%, LR: {lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    if best_model:
        model.load_state_dict(best_model)
        model = model.to(device)
    return model, best_acc


def main():
    random.seed(42)
    torch.manual_seed(42)

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    raw_clips_dir = datasets_dir / "raw_clips"
    video_crops_dir = script_dir / "video_crops_robust"
    models_dir = project_dir / "models"
    split_file = datasets_dir / "splits.json"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load splits
    with open(split_file) as f:
        splits = json.load(f)

    # Load MegaDetector
    print("\n" + "=" * 60)
    print("STEP 1: Loading MegaDetector")
    print("=" * 60)
    detector = pw_detection.MegaDetectorV6(
        device=device, pretrained=True, version="MDV6-yolov10-c"
    )

    # Extract crops
    print("\n" + "=" * 60)
    print("STEP 2: Extracting crops (more per clip)")
    print("=" * 60)

    import shutil
    if video_crops_dir.exists():
        shutil.rmtree(video_crops_dir)

    video_crops, video_labels = extract_crops_from_videos(
        raw_clips_dir, splits["train"], video_crops_dir, detector,
        sample_fps=4.0,  # Higher FPS for more diversity
        detection_conf=0.2,  # Lower threshold
        crops_per_clip=80   # More crops per clip
    )

    print(f"\nTotal crops: {len(video_crops)}")

    # Count per species
    label_counts = Counter(video_labels)
    print("Crops per species:")
    for species in SPECIES_CLASSES:
        idx = SPECIES_TO_IDX[species]
        print(f"  {species}: {label_counts.get(idx, 0)}")

    # Create train/val split
    print("\n" + "=" * 60)
    print("STEP 3: Creating train/val split")
    print("=" * 60)

    combined = list(zip(video_crops, video_labels))
    random.shuffle(combined)
    video_crops, video_labels = zip(*combined)
    video_crops, video_labels = list(video_crops), list(video_labels)

    val_split_idx = int(0.2 * len(video_crops))
    val_crops = video_crops[:val_split_idx]
    val_labels = video_labels[:val_split_idx]
    train_crops = video_crops[val_split_idx:]
    train_labels = video_labels[val_split_idx:]

    print(f"Train: {len(train_crops)}, Val: {len(val_crops)}")

    # Heavy augmentation for training
    print("\n" + "=" * 60)
    print("STEP 4: Setting up data loaders with heavy augmentation")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CroppedWildlifeDataset(train_crops, train_labels, train_transform)
    val_dataset = CroppedWildlifeDataset(val_crops, val_labels, val_transform)

    # Weighted sampling
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=24, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Train
    print("\n" + "=" * 60)
    print("STEP 5: Training ResNet50 with Focal Loss + MixUp")
    print("=" * 60)

    model, best_acc = train_classifier(
        train_loader, val_loader,
        num_classes=len(SPECIES_CLASSES),
        device=device,
        epochs=60,
        use_mixup=True
    )

    # Save
    print("\n" + "=" * 60)
    print("STEP 6: Saving model")
    print("=" * 60)

    model_path = models_dir / "species_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': SPECIES_CLASSES,
        'best_val_acc': best_acc,
        'model_type': 'resnet50'
    }, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Best validation accuracy: {best_acc:.1f}%")

    print(f"\nNext: python research/evaluate_twostage_tta.py")

    return 0


if __name__ == "__main__":
    exit(main())
