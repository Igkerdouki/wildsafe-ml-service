#!/usr/bin/env python3
"""
Evaluate the trained CLIP classifier on held-out test clips.

This script uses:
1. MegaDetector to detect animals in video frames
2. CLIP to extract embeddings from crops
3. Trained MLP classifier to predict species
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PytorchWildlife.models import detection as pw_detection
from transformers import CLIPProcessor, CLIPModel


SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]
IDX_TO_SPECIES = {idx: name for idx, name in enumerate(SPECIES_CLASSES)}


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


class CLIPClassifierEvaluator:
    """Evaluator using MegaDetector + CLIP embeddings + trained classifier."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = device

        # Load MegaDetector
        print("Loading MegaDetector V6...")
        self.detector = pw_detection.MegaDetectorV6(
            device=device, pretrained=True, version="MDV6-yolov10-c"
        )

        # Load CLIP for embeddings
        print("Loading CLIP ViT-B/32...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()

        # Load trained classifier
        print(f"Loading classifier from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        self.classifier = CLIPClassifier(
            input_dim=checkpoint.get('input_dim', 512),
            hidden_dim=256,
            num_classes=len(SPECIES_CLASSES),
            dropout=0.0  # No dropout at inference
        ).to(device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()

        print(f"Loaded model with {checkpoint.get('best_val_acc', 'N/A')}% validation accuracy")

    def extract_embedding(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Extract CLIP embedding from a crop."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        with torch.no_grad():
            inputs = self.clip_processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
            embedding = vision_outputs.pooler_output
            embedding = self.clip_model.visual_projection(embedding)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding

    def classify_crop(self, crop_bgr: np.ndarray) -> tuple:
        """Classify a crop using CLIP embedding + trained classifier."""
        embedding = self.extract_embedding(crop_bgr)

        with torch.no_grad():
            logits = self.classifier(embedding)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).item()
            confidence = probs[0, pred_idx].item()

        predicted_species = IDX_TO_SPECIES[pred_idx]

        # Get all probabilities
        all_probs = {IDX_TO_SPECIES[i]: probs[0, i].item() for i in range(len(SPECIES_CLASSES))}

        return predicted_species, confidence, all_probs

    def evaluate_video(
        self,
        video_path: Path,
        ground_truth_species: str,
        sample_fps: float = 5.0,
        detection_conf: float = 0.2,
        min_crop_size: int = 64
    ) -> dict:
        """Evaluate a video clip."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Could not open {video_path}"}

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30
        frame_interval = max(1, int(video_fps / sample_fps))

        # Accumulate weighted votes
        species_scores = defaultdict(float)
        species_counts = defaultdict(int)

        frame_count = 0
        total_detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect animals with MegaDetector
                results = self.detector.single_image_detection(
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
                            species, conf, _ = self.classify_crop(crop)

                            # Weight by confidence
                            species_scores[species] += conf
                            species_counts[species] += 1
                            total_detections += 1

            frame_count += 1

        cap.release()

        # Determine prediction
        if species_scores:
            predicted = max(species_scores.keys(), key=lambda s: species_scores[s])
            sorted_species = sorted(
                species_scores.keys(),
                key=lambda s: species_scores[s],
                reverse=True
            )[:3]
            top3_correct = ground_truth_species in sorted_species
        else:
            predicted = "none"
            sorted_species = []
            top3_correct = False

        is_correct = predicted == ground_truth_species

        return {
            "ground_truth": ground_truth_species,
            "predicted": predicted,
            "correct": is_correct,
            "detections": total_detections,
            "weighted_scores": {s: round(species_scores[s], 3) for s in sorted_species},
            "vote_counts": dict(species_counts),
            "top3": sorted_species,
            "top3_correct": top3_correct
        }


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    datasets_dir = script_dir / "datasets"
    split_file = datasets_dir / "splits.json"
    raw_clips_dir = datasets_dir / "raw_clips"
    model_path = project_dir / "models" / "clip_species_classifier.pt"

    if not split_file.exists():
        print(f"Error: Split file not found at {split_file}")
        return 1

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_clip_classifier.py first to train the model.")
        return 1

    with open(split_file) as f:
        splits = json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize evaluator
    evaluator = CLIPClassifierEvaluator(model_path, device=device)

    print("\n" + "=" * 60)
    print("TRAINED CLIP CLASSIFIER EVALUATION")
    print("=" * 60)

    results = {
        "total_clips": 0,
        "clips_correct": 0,
        "clips_top3_correct": 0,
        "per_species": {s: {"clips": 0, "correct": 0, "top3_correct": 0} for s in SPECIES_CLASSES},
        "clip_details": []
    }

    for species, clip_names in splits["test"].items():
        species_dir = raw_clips_dir / species
        for clip_name in clip_names:
            clip_path = species_dir / clip_name
            if not clip_path.exists():
                print(f"  Warning: {clip_path} not found")
                continue

            print(f"\n  [{species}] {clip_name}")

            clip_result = evaluator.evaluate_video(clip_path, species)

            if "error" in clip_result:
                print(f"    ERROR: {clip_result['error']}")
                continue

            status = "CORRECT" if clip_result["correct"] else f"WRONG -> {clip_result['predicted']}"
            print(f"    Result: {status}")
            print(f"    Detections: {clip_result['detections']}")
            print(f"    Top-3: {clip_result['top3']}")
            print(f"    Scores: {clip_result['weighted_scores']}")

            results["total_clips"] += 1
            results["per_species"][species]["clips"] += 1

            if clip_result["correct"]:
                results["clips_correct"] += 1
                results["per_species"][species]["correct"] += 1

            if clip_result["top3_correct"]:
                results["clips_top3_correct"] += 1
                results["per_species"][species]["top3_correct"] += 1

            results["clip_details"].append({
                "species": species,
                "clip": clip_name,
                **clip_result
            })

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Trained CLIP Classifier)")
    print("=" * 60)

    print(f"\nTest clips: {results['total_clips']}")
    print(f"Correctly classified: {results['clips_correct']}")
    print(f"Ground truth in top-3: {results['clips_top3_correct']}")

    if results['total_clips'] > 0:
        acc = 100 * results['clips_correct'] / results['total_clips']
        top3_acc = 100 * results['clips_top3_correct'] / results['total_clips']
        print(f"\nTop-1 accuracy: {acc:.1f}%")
        print(f"Top-3 accuracy: {top3_acc:.1f}%")

    print("\nPer-species results:")
    print("-" * 60)
    print(f"{'Species':<12} {'Clips':<8} {'Correct':<10} {'Top-3':<10} {'Acc':<10}")
    print("-" * 60)

    for species in SPECIES_CLASSES:
        data = results["per_species"][species]
        if data["clips"] > 0:
            acc = 100 * data["correct"] / data["clips"]
            print(f"{species:<12} {data['clips']:<8} {data['correct']:<10} {data['top3_correct']:<10} {acc:.0f}%")

    print("-" * 60)

    # Save results
    output_file = script_dir / "evaluation_clip_classifier_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
