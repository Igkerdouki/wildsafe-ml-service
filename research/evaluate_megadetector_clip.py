#!/usr/bin/env python3
"""
MegaDetector + CLIP Ensemble for Wildlife Species Classification.

This pipeline uses:
1. MegaDetector V6 for robust animal detection (any animal)
2. CLIP for zero-shot species classification (no training needed)

CLIP is pre-trained on 400M image-text pairs and generalizes well.
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
from PIL import Image
from PytorchWildlife.models import detection as pw_detection

# Try to import CLIP from transformers
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")


SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]

# Optimized CLIP prompts - more descriptive for better matching
SPECIES_PROMPTS = {
    "bear": [
        "a photograph of a bear walking",
        "a black bear in the wild",
        "a grizzly bear",
        "a brown bear in nature",
    ],
    "coyote": [
        "a photograph of a coyote",
        "a wild coyote walking",
        "a coyote in grassland",
        "a coyote hunting",
    ],
    "deer": [
        "a photograph of a deer",
        "a white-tailed deer standing",
        "a deer in the forest",
        "a doe or buck deer",
    ],
    "elk": [
        "a photograph of an elk",
        "a bull elk with antlers",
        "an elk in a meadow",
        "a large elk standing",
    ],
    "fox": [
        "a photograph of a fox",
        "a red fox in nature",
        "a fox walking",
        "a wild fox",
    ],
    "goat": [
        "a photograph of a mountain goat",
        "a wild goat on rocks",
        "a goat in the mountains",
        "a billy goat",
    ],
    "horse": [
        "a photograph of a horse",
        "a wild horse running",
        "a mustang horse",
        "a horse in a field",
    ],
    "moose": [
        "a photograph of a moose",
        "a bull moose with large antlers",
        "a moose in the wild",
        "a large moose standing",
    ],
    "opossum": [
        "a photograph of an opossum",
        "a virginia opossum",
        "an opossum at night",
        "a possum walking",
    ],
    "raccoon": [
        "a photograph of a raccoon",
        "a raccoon foraging",
        "a raccoon with mask markings",
        "a wild raccoon",
    ],
    "skunk": [
        "a photograph of a skunk",
        "a striped skunk",
        "a black and white skunk",
        "a skunk in nature",
    ],
    "wild_boar": [
        "a photograph of a wild boar",
        "a feral pig",
        "a wild pig with tusks",
        "a boar in the forest",
    ],
}


class MegaDetectorCLIPEnsemble:
    """Two-stage pipeline: MegaDetector detection + CLIP classification."""

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Load MegaDetector
        print("Loading MegaDetector V6...")
        self.detector = pw_detection.MegaDetectorV6(
            device=device, pretrained=True, version="MDV6-yolov10-c"
        )

        # Load CLIP
        if CLIP_AVAILABLE:
            print("Loading CLIP ViT-B/32...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(device)
            self.clip_model.eval()

            # Pre-compute text embeddings
            print("Pre-computing text embeddings...")
            self.text_embeddings = self._compute_text_embeddings()
        else:
            self.clip_model = None
            self.clip_processor = None
            self.text_embeddings = None

    def _compute_text_embeddings(self):
        """Pre-compute CLIP text embeddings for all species."""
        embeddings = {}

        with torch.no_grad():
            for species, prompts in SPECIES_PROMPTS.items():
                inputs = self.clip_processor(
                    text=prompts, return_tensors="pt", padding=True
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Use text_model directly to get embeddings
                text_outputs = self.clip_model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # Get pooled output and project through text_projection
                text_features = text_outputs.pooler_output
                text_features = self.clip_model.text_projection(text_features)

                # Average across prompts and normalize
                text_features = text_features.mean(dim=0, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings[species] = text_features

        return embeddings

    def classify_crop(self, crop_bgr: np.ndarray) -> tuple:
        """Classify a crop using CLIP zero-shot."""
        if self.clip_model is None:
            return "unknown", 0.0, {}

        # Convert BGR to RGB PIL
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        with torch.no_grad():
            inputs = self.clip_processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # Use vision_model directly to get embeddings
            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
            image_features = vision_outputs.pooler_output
            image_features = self.clip_model.visual_projection(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity with all species
            similarities = {}
            for species, text_emb in self.text_embeddings.items():
                sim = (image_features @ text_emb.T).squeeze().item()
                similarities[species] = sim

        # Get best match
        best_species = max(similarities.keys(), key=lambda s: similarities[s])
        confidence = similarities[best_species]

        return best_species, confidence, similarities

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

                            # Weight by similarity (CLIP sims are typically 0.2-0.4)
                            weight = max(0, conf)
                            species_scores[species] += weight
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

    # Initialize ensemble
    ensemble = MegaDetectorCLIPEnsemble(device=device)

    print("\n" + "=" * 60)
    print("MEGADETECTOR + CLIP ENSEMBLE EVALUATION")
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

            clip_result = ensemble.evaluate_video(clip_path, species)

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
    print("RESULTS SUMMARY (MegaDetector + CLIP)")
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
    output_file = script_dir / "evaluation_megadetector_clip_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
