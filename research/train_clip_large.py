#!/usr/bin/env python3
"""
Train an improved classifier using CLIP ViT-L/14 (larger model).

Key improvements over base CLIP classifier:
1. Uses ViT-L/14 instead of ViT-B/32 (4x more parameters)
2. Larger hidden dimensions in MLP
3. More aggressive data augmentation during embedding extraction
4. Class-balanced sampling
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from PytorchWildlife.models import detection as pw_detection
from transformers import CLIPProcessor, CLIPModel

SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_CLASSES)}


class CLIPClassifierLarge(nn.Module):
    """Larger MLP classifier for CLIP ViT-L/14 embeddings (768-dim)."""

    def __init__(self, input_dim=768, hidden_dim=512, num_classes=12, dropout=0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def random_crop_resize(img, scale_range=(0.8, 1.0)):
    """Random crop and resize for augmentation."""
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)

    if new_h < h and new_w < w:
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        img = img[top:top+new_h, left:left+new_w]

    return cv2.resize(img, (224, 224))


def extract_clip_embeddings(
    clip_model,
    clip_processor,
    detector,
    raw_clips_dir: Path,
    clips_dict: dict,
    device: str,
    sample_fps: float = 5.0,
    detection_conf: float = 0.15,
    min_crop_size: int = 48,
    max_crops_per_clip: int = 80,
    augment: bool = True
):
    """Extract CLIP ViT-L/14 embeddings with augmentation."""
    all_embeddings = []
    all_labels = []

    for species, clip_names in clips_dict.items():
        species_dir = raw_clips_dir / species
        species_idx = SPECIES_TO_IDX[species]

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

            clip_embeddings = []
            frame_count = 0

            while len(clip_embeddings) < max_crops_per_clip:
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

                                    # Expand box slightly for context
                                    pad_x = int((x2 - x1) * 0.1)
                                    pad_y = int((y2 - y1) * 0.1)
                                    x1 = max(0, x1 - pad_x)
                                    y1 = max(0, y1 - pad_y)
                                    x2 = min(w, x2 + pad_x)
                                    y2 = min(h, y2 + pad_y)

                                    if x2 - x1 < min_crop_size or y2 - y1 < min_crop_size:
                                        continue

                                    crop = frame[y1:y2, x1:x2]
                                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                                    # Extract embedding for original
                                    pil_img = Image.fromarray(crop_rgb)
                                    with torch.no_grad():
                                        inputs = clip_processor(
                                            images=pil_img, return_tensors="pt"
                                        )
                                        pixel_values = inputs["pixel_values"].to(device)
                                        outputs = clip_model.get_image_features(pixel_values)
                                        embedding = outputs / outputs.norm(dim=-1, keepdim=True)
                                    clip_embeddings.append(embedding.cpu())

                                    # Augmented versions (if training)
                                    if augment and len(clip_embeddings) < max_crops_per_clip:
                                        # Horizontal flip
                                        flipped = cv2.flip(crop_rgb, 1)
                                        pil_flip = Image.fromarray(flipped)
                                        with torch.no_grad():
                                            inputs = clip_processor(
                                                images=pil_flip, return_tensors="pt"
                                            )
                                            pixel_values = inputs["pixel_values"].to(device)
                                            outputs = clip_model.get_image_features(pixel_values)
                                            embedding = outputs / outputs.norm(dim=-1, keepdim=True)
                                        clip_embeddings.append(embedding.cpu())

                                    if len(clip_embeddings) >= max_crops_per_clip:
                                        break
                    except Exception as e:
                        pass

                frame_count += 1

            cap.release()
            print(f"{len(clip_embeddings)} embeddings")

            for emb in clip_embeddings:
                all_embeddings.append(emb)
                all_labels.append(species_idx)

    return all_embeddings, all_labels


def train_classifier(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    device,
    epochs=150,
    batch_size=64,
    lr=0.0003
):
    """Train the MLP classifier on CLIP embeddings with class balancing."""
    X_train = torch.cat(train_embeddings, dim=0)
    y_train = torch.tensor(train_labels)
    X_val = torch.cat(val_embeddings, dim=0)
    y_val = torch.tensor(val_labels)

    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")

    # Class-balanced sampling
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create classifier
    model = CLIPClassifierLarge(
        input_dim=X_train.shape[1],
        hidden_dim=512,
        num_classes=len(SPECIES_CLASSES),
        dropout=0.4
    ).to(device)

    # Loss with class weights
    class_weights_tensor = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor.to(device),
        label_smoothing=0.1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    best_acc = 0
    best_model = None
    patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

                for pred, label in zip(predicted.cpu(), y_batch.cpu()):
                    per_class_total[label.item()] += 1
                    if pred == label:
                        per_class_correct[label.item()] += 1

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train: {train_acc:.1f}%, Val: {val_acc:.1f}%")
            # Show per-class accuracy
            if (epoch + 1) % 50 == 0:
                print("  Per-class validation accuracy:")
                for cls_idx in sorted(per_class_total.keys()):
                    cls_name = SPECIES_CLASSES[cls_idx]
                    cls_acc = 100 * per_class_correct[cls_idx] / per_class_total[cls_idx]
                    print(f"    {cls_name}: {cls_acc:.1f}%")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model:
        model.load_state_dict(best_model)
        model = model.to(device)

    return model, best_acc


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    raw_clips_dir = datasets_dir / "raw_clips"
    models_dir = project_dir / "models"
    split_file = datasets_dir / "splits.json"

    if not split_file.exists():
        print(f"Error: Split file not found at {split_file}")
        return 1

    with open(split_file) as f:
        splits = json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    print("\n" + "=" * 60)
    print("STEP 1: Loading models")
    print("=" * 60)

    print("Loading MegaDetector V6...")
    detector = pw_detection.MegaDetectorV6(
        device=device, pretrained=True, version="MDV6-yolov10-c"
    )

    print("Loading CLIP ViT-L/14 (large model)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Extract embeddings from training videos
    print("\n" + "=" * 60)
    print("STEP 2: Extracting CLIP ViT-L/14 embeddings from training videos")
    print("=" * 60)

    embeddings, labels = extract_clip_embeddings(
        clip_model, clip_processor, detector,
        raw_clips_dir, splits["train"], device,
        sample_fps=6.0,
        max_crops_per_clip=100,
        augment=True
    )

    print(f"\nTotal embeddings: {len(embeddings)}")

    # Count per species
    label_counts = Counter(labels)
    print("Embeddings per species:")
    for species in SPECIES_CLASSES:
        idx = SPECIES_TO_IDX[species]
        print(f"  {species}: {label_counts.get(idx, 0)}")

    # Split into train/val
    print("\n" + "=" * 60)
    print("STEP 3: Creating train/val split (stratified)")
    print("=" * 60)

    # Stratified split
    indices_by_class = defaultdict(list)
    for i, label in enumerate(labels):
        indices_by_class[label].append(i)

    train_indices = []
    val_indices = []

    for cls_idx, indices in indices_by_class.items():
        random.shuffle(indices)
        val_count = max(1, int(0.2 * len(indices)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    train_emb = [embeddings[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_emb = [embeddings[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f"Train: {len(train_emb)}, Val: {len(val_emb)}")

    # Train classifier
    print("\n" + "=" * 60)
    print("STEP 4: Training larger MLP classifier on CLIP ViT-L/14 embeddings")
    print("=" * 60)

    classifier, best_acc = train_classifier(
        train_emb, train_labels,
        val_emb, val_labels,
        device,
        epochs=200,
        batch_size=64,
        lr=0.0005
    )

    # Save model
    print("\n" + "=" * 60)
    print("STEP 5: Saving model")
    print("=" * 60)

    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "clip_large_classifier.pt"
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'classes': SPECIES_CLASSES,
        'best_val_acc': best_acc,
        'input_dim': 768,  # ViT-L/14 output dimension
        'model_type': 'clip_vit_l_14_mlp'
    }, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Best validation accuracy: {best_acc:.1f}%")

    print("\nNext: python research/evaluate_clip_large.py")

    return 0


if __name__ == "__main__":
    exit(main())
