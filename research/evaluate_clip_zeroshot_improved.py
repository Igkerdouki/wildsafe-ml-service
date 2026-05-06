#!/usr/bin/env python3
"""
Evaluate CLIP zero-shot classification on wildlife videos.

This approach uses CLIP's zero-shot capability directly without MegaDetector.
We sample frames and classify each frame using text prompts.
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


SPECIES_CLASSES = [
    "bear", "coyote", "deer", "elk", "fox", "goat",
    "horse", "moose", "opossum", "raccoon", "skunk", "wild_boar"
]

# Enhanced text prompts for better zero-shot performance
TEXT_PROMPTS = {
    "bear": ["a photo of a bear", "a black bear in nature", "a grizzly bear", "a bear walking"],
    "coyote": ["a photo of a coyote", "a coyote in the wild", "a wild coyote", "a coyote with pointed ears"],
    "deer": ["a photo of a deer", "a white-tailed deer", "a deer in nature", "a buck deer"],
    "elk": ["a photo of an elk", "an elk with antlers", "a bull elk", "an elk in a field"],
    "fox": ["a photo of a fox", "a red fox", "a fox in nature", "a wild fox"],
    "goat": ["a photo of a goat", "a mountain goat", "a wild goat", "a goat on rocks"],
    "horse": ["a photo of a horse", "a wild horse", "a mustang horse", "horses in a field"],
    "moose": ["a photo of a moose", "a bull moose with large antlers", "a moose with palmate antlers", "a dark brown moose"],
    "opossum": ["a photo of an opossum", "a virginia opossum", "an opossum at night"],
    "raccoon": ["a photo of a raccoon", "a raccoon with bandit mask", "a wild raccoon"],
    "skunk": ["a photo of a skunk", "a striped skunk", "a skunk with stripe"],
    "wild_boar": ["a photo of a wild boar", "a wild pig", "a feral hog", "a boar in forest"]
}


class CLIPZeroShotEvaluator:
    """Zero-shot classifier using CLIP directly."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cpu"):
        self.device = device

        print(f"Loading CLIP {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Pre-compute text embeddings for all prompts
        print("Computing text embeddings...")
        self.species_embeddings = {}

        with torch.no_grad():
            for species, prompts in TEXT_PROMPTS.items():
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                text_outputs = self.model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                # Access pooler_output from the model output
                text_features = text_outputs.pooler_output
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Average embeddings for all prompts of this species
                self.species_embeddings[species] = text_features.mean(dim=0)

        # Stack into a single tensor for efficient batch processing
        self.text_embeddings = torch.stack([
            self.species_embeddings[s] for s in SPECIES_CLASSES
        ])
        print("Ready!")

    def classify_frame(self, frame_rgb: np.ndarray) -> dict:
        """Classify a single frame."""
        pil_img = Image.fromarray(frame_rgb)

        with torch.no_grad():
            inputs = self.processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # Get image embedding
            image_outputs = self.model.get_image_features(pixel_values)
            image_features = image_outputs.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity to all species
            similarity = (image_features @ self.text_embeddings.T).squeeze()
            probs = torch.softmax(similarity * 100, dim=0)  # Temperature scaling

        return {
            SPECIES_CLASSES[i]: probs[i].item()
            for i in range(len(SPECIES_CLASSES))
        }

    def evaluate_video(
        self,
        video_path: Path,
        ground_truth_species: str,
        sample_fps: float = 3.0,
        min_confidence: float = 0.1
    ) -> dict:
        """Evaluate a video clip using frame-level classification."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Could not open {video_path}"}

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30
        frame_interval = max(1, int(video_fps / sample_fps))

        # Accumulate scores per species
        species_scores = defaultdict(float)
        species_counts = defaultdict(int)
        frames_processed = 0

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Classify full frame
                probs = self.classify_frame(frame_rgb)

                # Accumulate weighted scores
                for species, prob in probs.items():
                    if prob > min_confidence:
                        species_scores[species] += prob
                        species_counts[species] += 1

                frames_processed += 1

            frame_count += 1

        cap.release()

        # Determine prediction
        if species_scores:
            # Normalize by count to get average confidence
            avg_scores = {
                s: species_scores[s] / max(1, species_counts[s])
                for s in species_scores
            }

            # Weight by both average confidence and vote count
            weighted_scores = {
                s: avg_scores[s] * np.log1p(species_counts[s])
                for s in species_scores
            }

            predicted = max(weighted_scores.keys(), key=lambda s: weighted_scores[s])
            sorted_species = sorted(
                weighted_scores.keys(),
                key=lambda s: weighted_scores[s],
                reverse=True
            )[:3]
            top3_correct = ground_truth_species in sorted_species
        else:
            predicted = "none"
            sorted_species = []
            top3_correct = False
            avg_scores = {}
            weighted_scores = {}

        is_correct = predicted == ground_truth_species

        return {
            "ground_truth": ground_truth_species,
            "predicted": predicted,
            "correct": is_correct,
            "frames_processed": frames_processed,
            "avg_scores": {s: round(v, 4) for s, v in sorted(avg_scores.items(), key=lambda x: -x[1])[:5]},
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

    # Initialize evaluator with large CLIP model
    evaluator = CLIPZeroShotEvaluator(
        model_name="openai/clip-vit-large-patch14",
        device=device
    )

    print("\n" + "=" * 60)
    print("CLIP ZERO-SHOT EVALUATION (ViT-L/14)")
    print("=" * 60)

    results = {
        "total_clips": 0,
        "clips_correct": 0,
        "clips_top3_correct": 0,
        "model": "clip-vit-large-patch14",
        "method": "zero-shot",
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
            print(f"    Frames: {clip_result['frames_processed']}")
            print(f"    Top-3: {clip_result['top3']}")
            print(f"    Avg scores: {clip_result['avg_scores']}")

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
    print("RESULTS SUMMARY (CLIP Zero-Shot ViT-L/14)")
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
    output_file = script_dir / "evaluation_clip_zeroshot_large_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
