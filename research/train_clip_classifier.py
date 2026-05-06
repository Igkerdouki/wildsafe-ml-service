#!/usr/bin/env python3
"""
Train a classifier on top of CLIP embeddings.

This approach:
1. Uses MegaDetector to extract animal crops from training videos
2. Extracts CLIP embeddings for each crop (frozen, no training)
3. Trains a small MLP classifier on top of the embeddings

Benefits:
- Leverages CLIP's pre-trained knowledge (400M images)
- Much faster training (only trains small MLP)
- Better generalization than training from scratch
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
from torch.utils.data import DataLoader, TensorDataset
from PytorchWildlife.models import detection as pw_detection
from transformers import CLIPProcessor, CLIPModel

SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_CLASSES)}


class CLIPClassifier(nn.Module):
    """Small MLP classifier on top of CLIP embeddings."""

    def __init__(self, input_dim=512, hidden_dim=256, num_classes=12, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def extract_clip_embeddings(
    clip_model,
    clip_processor,
    detector,
    raw_clips_dir: Path,
    clips_dict: dict,
    device: str,
    sample_fps: float = 3.0,
    detection_conf: float = 0.2,
    min_crop_size: int = 64,
    max_crops_per_clip: int = 50
):
    """Extract CLIP embeddings for all animal crops in training videos."""
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
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w, x2), min(h, y2)

                                    if x2 - x1 < min_crop_size or y2 - y1 < min_crop_size:
                                        continue

                                    crop = frame[y1:y2, x1:x2]
                                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    pil_img = Image.fromarray(crop_rgb)

                                    # Extract CLIP embedding
                                    with torch.no_grad():
                                        inputs = clip_processor(
                                            images=pil_img, return_tensors="pt"
                                        )
                                        pixel_values = inputs["pixel_values"].to(device)

                                        vision_outputs = clip_model.vision_model(
                                            pixel_values=pixel_values
                                        )
                                        embedding = vision_outputs.pooler_output
                                        embedding = clip_model.visual_projection(embedding)
                                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                                    clip_embeddings.append(embedding.cpu())

                                    if len(clip_embeddings) >= max_crops_per_clip:
                                        break
                    except Exception:
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
    epochs=100,
    batch_size=64,
    lr=0.001
):
    """Train the MLP classifier on CLIP embeddings."""
    # Stack embeddings
    X_train = torch.cat(train_embeddings, dim=0)
    y_train = torch.tensor(train_labels)
    X_val = torch.cat(val_embeddings, dim=0)
    y_val = torch.tensor(val_labels)

    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create classifier with stronger regularization
    model = CLIPClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=256,
        num_classes=len(SPECIES_CLASSES),
        dropout=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    best_model = None

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
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train: {train_acc:.1f}%, Val: {val_acc:.1f}%")

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

    print("Loading CLIP ViT-B/32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Extract embeddings from training videos
    print("\n" + "=" * 60)
    print("STEP 2: Extracting CLIP embeddings from training videos")
    print("=" * 60)

    embeddings, labels = extract_clip_embeddings(
        clip_model, clip_processor, detector,
        raw_clips_dir, splits["train"], device,
        sample_fps=5.0,
        max_crops_per_clip=100
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
    print("STEP 3: Creating train/val split")
    print("=" * 60)

    combined = list(zip(embeddings, labels))
    random.shuffle(combined)
    embeddings, labels = zip(*combined)
    embeddings, labels = list(embeddings), list(labels)

    val_split = int(0.2 * len(embeddings))
    val_emb = embeddings[:val_split]
    val_labels = labels[:val_split]
    train_emb = embeddings[val_split:]
    train_labels = labels[val_split:]

    print(f"Train: {len(train_emb)}, Val: {len(val_emb)}")

    # Train classifier
    print("\n" + "=" * 60)
    print("STEP 4: Training MLP classifier on CLIP embeddings")
    print("=" * 60)

    classifier, best_acc = train_classifier(
        train_emb, train_labels,
        val_emb, val_labels,
        device,
        epochs=150,
        batch_size=32,
        lr=0.0005
    )

    # Save model
    print("\n" + "=" * 60)
    print("STEP 5: Saving model")
    print("=" * 60)

    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "clip_species_classifier.pt"
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'classes': SPECIES_CLASSES,
        'best_val_acc': best_acc,
        'input_dim': 512,
        'model_type': 'clip_mlp'
    }, model_path)

    print(f"Model saved to: {model_path}")
    print(f"Best validation accuracy: {best_acc:.1f}%")

    print("\nNext: python research/evaluate_clip_classifier.py")

    return 0


if __name__ == "__main__":
    exit(main())
