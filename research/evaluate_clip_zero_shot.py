#!/usr/bin/env python3
"""
Zero-shot wildlife species classification using CLIP.

CLIP is trained on diverse web images and can generalize well to new domains
without any training data.

Usage:
    pip install transformers
    python research/evaluate_clip_zero_shot.py
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from PytorchWildlife.models import detection as pw_detection
import numpy as np


SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]

# CLIP text prompts for each species
SPECIES_PROMPTS = {
    "bear": ["a photo of a bear", "a black bear", "a grizzly bear", "a bear in the wild"],
    "coyote": ["a photo of a coyote", "a wild coyote", "a coyote in nature"],
    "deer": ["a photo of a deer", "a white-tailed deer", "a deer in the forest"],
    "elk": ["a photo of an elk", "an elk with antlers", "a bull elk"],
    "fox": ["a photo of a fox", "a red fox", "a fox in nature"],
    "goat": ["a photo of a mountain goat", "a wild goat", "a goat on rocks"],
    "horse": ["a photo of a horse", "a wild horse", "a mustang"],
    "moose": ["a photo of a moose", "a bull moose", "a moose in the wild"],
    "opossum": ["a photo of an opossum", "a virginia opossum", "an opossum at night"],
    "raccoon": ["a photo of a raccoon", "a raccoon foraging", "a wild raccoon"],
    "skunk": ["a photo of a skunk", "a striped skunk", "a skunk in nature"],
    "wild_boar": ["a photo of a wild boar", "a feral pig", "a wild pig"],
}


def load_clip_model(device):
    """Load CLIP model and processor."""
    print("Loading CLIP model (ViT-B/32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    return model, processor


def get_text_embeddings(model, processor, device):
    """Pre-compute text embeddings for all species prompts."""
    text_embeddings = {}

    with torch.no_grad():
        for species, prompts in SPECIES_PROMPTS.items():
            inputs = processor(text=prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Get text embeddings using forward pass
            text_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Average the embeddings for multiple prompts
            text_features = text_features.mean(dim=0, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embeddings[species] = text_features

    return text_embeddings


def classify_crop_clip(model, processor, crop_img, text_embeddings, device):
    """Classify a crop using CLIP zero-shot classification."""
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    with torch.no_grad():
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarity with all species
        similarities = {}
        for species, text_feat in text_embeddings.items():
            sim = (image_features @ text_feat.T).squeeze().item()
            similarities[species] = sim

    # Get best match
    best_species = max(similarities.keys(), key=lambda s: similarities[s])
    confidence = similarities[best_species]

    return best_species, confidence, similarities


def evaluate_clip_on_clip_video(
    video_path: Path,
    ground_truth_species: str,
    detector,
    clip_model,
    clip_processor,
    text_embeddings,
    device: str,
    sample_fps: float = 5.0,
    detection_conf: float = 0.2,
    min_crop_size: int = 64
):
    """Evaluate a video clip using CLIP zero-shot classification."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = max(1, int(video_fps / sample_fps))

    # Accumulate similarity-weighted votes
    species_scores = defaultdict(float)
    species_counts = defaultdict(int)

    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            results = detector.single_image_detection(frame_rgb, det_conf_thres=detection_conf)
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
                        species, conf, all_sims = classify_crop_clip(
                            clip_model, clip_processor, crop, text_embeddings, device
                        )

                        # Weight by confidence
                        weight = max(0, conf)  # CLIP similarities can be negative
                        species_scores[species] += weight
                        species_counts[species] += 1
                        total_detections += 1

        frame_count += 1

    cap.release()

    if species_scores:
        predicted_species = max(species_scores.keys(), key=lambda s: species_scores[s])
        sorted_species = sorted(
            species_scores.keys(),
            key=lambda s: species_scores[s],
            reverse=True
        )[:3]
        top3_contains_gt = ground_truth_species in sorted_species
    else:
        predicted_species = "none"
        sorted_species = []
        top3_contains_gt = False

    is_correct = predicted_species == ground_truth_species

    return {
        "ground_truth": ground_truth_species,
        "predicted": predicted_species,
        "correct": is_correct,
        "detections": total_detections,
        "weighted_scores": {s: round(species_scores[s], 3) for s in sorted_species},
        "vote_counts": dict(species_counts),
        "top3": sorted_species,
        "top3_correct": top3_contains_gt
    }


def main():
    script_dir = Path(__file__).parent
    datasets_dir = script_dir / "datasets"

    split_file = datasets_dir / "splits.json"
    raw_clips_dir = datasets_dir / "raw_clips"

    if not split_file.exists():
        print(f"Error: Split file not found at {split_file}")
        return 1

    with open(split_file) as f:
        splits = json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading MegaDetector...")
    detector = pw_detection.MegaDetectorV6(
        device=device, pretrained=True, version="MDV6-yolov10-c"
    )

    clip_model, clip_processor = load_clip_model(device)
    text_embeddings = get_text_embeddings(clip_model, clip_processor, device)

    print("\n" + "=" * 60)
    print("CLIP ZERO-SHOT SPECIES CLASSIFICATION")
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

            clip_result = evaluate_clip_on_clip_video(
                clip_path,
                species,
                detector,
                clip_model,
                clip_processor,
                text_embeddings,
                device
            )

            if "error" in clip_result:
                print(f"    ERROR: {clip_result['error']}")
                continue

            status = "CORRECT" if clip_result["correct"] else f"WRONG -> {clip_result['predicted']}"
            print(f"    Result: {status}")
            print(f"    Detections: {clip_result['detections']}")
            print(f"    Top-3: {clip_result['top3']}")
            print(f"    Weighted scores: {clip_result['weighted_scores']}")

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
    print("RESULTS SUMMARY (CLIP Zero-Shot)")
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

    # Save
    output_file = script_dir / "evaluation_clip_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
