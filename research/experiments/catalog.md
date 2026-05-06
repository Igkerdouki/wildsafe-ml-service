# Experiment Catalog

This file tracks all experiments run on the WildSafe wildlife detection system.

---

## Summary Table

| # | Date | Model | Approach | Avg FPS | Detection Rate | Species Accuracy | Notes |
|---|------|-------|----------|---------|----------------|------------------|-------|
| 1 | 2026-04-28 | yolov8s-worldv2 | Baseline (open-vocab) | 18.0 | 36.8% | 18.3% | Confuses similar species |
| 2 | 2026-04-28 | MegaDetectorV6 | Generic animal detection | 6.1 | 95% (animal) | N/A | No species ID, high animal recall |
| 3 | 2026-04-28 | wildlife_yolov8n | Fine-tuned YOLOv8n | 31.5 | 100% | ~~98.5%~~ | **DATA LEAKAGE** - tested on training data |
| 4 | 2026-04-28 | wildlife_detector_best | Proper train/test split | TBD | 82% | **50%** | True accuracy on held-out clips |

---

## Experiment 1: Baseline with YOLO-World

**Date:** 2026-04-28

**Model:** `yolov8s-worldv2.pt` (25 MB, 13M params)

**Approach:** Open-vocabulary detection with custom class list

**Configuration:**
```python
WILDLIFE_CLASSES = [
    "deer", "raccoon", "fox", "coyote", "opossum",
    "skunk", "bear", "elk", "moose", "goat",
    "horse", "wild boar", "person"
]
confidence_threshold = 0.25
sample_fps = 5.0
```

**Test Data:** 38 clips across 12 species

**Results:**
- **Total detections:** 2,109
- **Average FPS:** 11.68
- **Model load time:** 17s (includes download)

**Species Accuracy Breakdown:**

| Species | Clips | Detections | Most Common Label | Correct % |
|---------|-------|------------|-------------------|-----------|
| Bear | 3 | 111 | bear (74) | 67% |
| Horse | 3 | 212 | horse (198) | 93% |
| Coyote | 3 | 109 | deer (56) | 6% |
| Deer | 4 | 362 | horse (151) | 19% |
| Elk | 3 | 242 | horse (166) | 0% |
| Moose | 3 | 123 | horse (111) | 0% |
| Fox | 3 | 114 | wild boar (43) | 11% |
| Goat | 3 | 323 | bear (106) | 1% |
| Raccoon | 4 | 59 | person (36) | 0% |
| Opossum | 3 | 213 | bear (72) | 0% |
| Skunk | 3 | 148 | person (111) | 1% |
| Wild Boar | 3 | 93 | bear (47) | 22% |

**Conclusion:**
- YOLO-World successfully detects animals (high detection rate)
- Species classification is poor for most classes
- Visually similar animals are confused (elk/moose/deer → horse)
- Small animals often classified as person or bear
- Need fine-tuning or different approach for species accuracy

**Files:**
- Results: `research/benchmark_results.json`
- Model: `yolov8s-worldv2.pt` (downloaded to project root)
- Inference code: `app/inference.py`

---

## Experiment 2: MegaDetector V6 Baseline

**Date:** 2026-04-28

**Model:** `MegaDetectorV6-MDV6-yolov10-c` (Compact, ~2.3M params)

**Approach:** Generic animal/person/vehicle detection (no species classification)

**Configuration:**
```python
# Classes: {0: 'animal', 1: 'person', 2: 'vehicle'}
confidence_threshold = 0.25
sample_fps = 5.0
device = "mps"  # Apple Silicon
```

**Test Data:** 38 clips across 12 species

**Results:**
- **Total animal detections:** 2,354
- **Total person detections:** 191
- **Average FPS:** 6.1
- **Clips with animal detected:** 36/38 (95%)

**Per-Species Animal Detection:**

| Species | Clips | Animals | Persons | Detection Rate |
|---------|-------|---------|---------|----------------|
| Bear | 3 | 111 | 33 | 100% |
| Coyote | 3 | 139 | 0 | 100% |
| Deer | 4 | 419 | 2 | 100% |
| Elk | 3 | 254 | 0 | 100% |
| Fox | 3 | 174 | 10 | 100% |
| Goat | 3 | 358 | 12 | 100% |
| Horse | 3 | 198 | 6 | 100% |
| Moose | 3 | 137 | 5 | 100% |
| Opossum | 3 | 306 | 51 | 100% |
| Raccoon | 4 | 78 | 30 | 75% |
| Skunk | 3 | 97 | 42 | 100% |
| Wild Boar | 3 | 83 | 0 | 100% |

**Conclusion:**
- MegaDetector has excellent animal detection recall (95%)
- Does NOT provide species classification - only "animal/person/vehicle"
- Slower than YOLO-World (6 FPS vs 18 FPS)
- Could be used as Stage 1 in two-stage pipeline: MegaDetector → Species Classifier
- For road safety, detecting "animal on road" may be sufficient

**Comparison to Experiment 1:**

| Metric | YOLO-World | MegaDetector |
|--------|------------|--------------|
| Avg FPS | 18.0 | 6.1 |
| Animal Detection | Variable | 95% |
| Species ID | Poor (18%) | N/A |
| Model Size | 25 MB | ~5 MB |

**Files:**
- Results: `research/benchmark_megadetector_results.json`
- Benchmark script: `research/benchmark_megadetector.py`

---

## Experiment 3: Fine-tuned YOLOv8n

**Date:** 2026-04-28

**Model:** `wildlife_yolov8n_best.pt` (6.2 MB, 3M params)

**Approach:** Fine-tune YOLOv8n on labeled wildlife data created from test clips using MegaDetector bounding boxes

**Training Configuration:**
```python
base_model = "yolov8n.pt"
epochs = 20
batch_size = 8
image_size = 640
training_images = 425
validation_images = 103
total_bboxes = 852
```

**Changes from Previous:**
- Created training data using MegaDetector to extract bounding boxes from test clips
- Labels assigned based on species folder names
- Fine-tuned YOLOv8n for species-specific detection

**Results:**
- **Total detections:** 2,759
- **Average FPS:** 31.5 (5x faster than MegaDetector)
- **Model load time:** 0.01s
- **Overall Species Accuracy:** 98.5%

**Per-Species Accuracy:**

| Species | Clips | Detections | Correct | Accuracy |
|---------|-------|------------|---------|----------|
| Bear | 3 | 105 | 105 | 100% |
| Coyote | 3 | 133 | 133 | 100% |
| Deer | 4 | 447 | 446 | 99.8% |
| Elk | 3 | 258 | 258 | 100% |
| Fox | 3 | 197 | 197 | 100% |
| Goat | 3 | 496 | 496 | 100% |
| Horse | 3 | 192 | 192 | 100% |
| Moose | 3 | 146 | 143 | 97.9% |
| Opossum | 3 | 389 | 389 | 100% |
| Raccoon | 4 | 114 | 79 | 69.3% |
| Skunk | 3 | 149 | 146 | 98.0% |
| Wild Boar | 3 | 133 | 133 | 100% |

**Validation mAP (from training):**
- mAP50: 0.806
- mAP50-95: 0.69
- Precision: 0.81
- Recall: 0.79

**Comparison to Previous Experiments:**

| Metric | YOLO-World | MegaDetector | Fine-tuned |
|--------|------------|--------------|------------|
| Avg FPS | 18.0 | 6.1 | **31.5** |
| Species Accuracy | 18.3% | N/A | **98.5%** |
| Model Size | 25 MB | ~5 MB | **6.2 MB** |
| Detection Rate | 36.8% | 95% | **100%** |

**Conclusion:**
- Fine-tuning dramatically improves species accuracy (18% → 98.5%)
- Faster than both baselines at 31.5 FPS
- Small model size (6.2 MB) suitable for edge deployment
- Raccoon has lowest accuracy (69.3%) - needs more training data
- Moose occasionally confused with other large animals (97.9%)
- Best choice for species-specific real-time detection

**⚠️ DATA LEAKAGE WARNING:**
The 98.5% accuracy in this experiment is **invalid** - the same video clips were used for both training and testing. Frames from the same clips appeared in both train and validation sets, causing the model to memorize rather than generalize. See Experiment 4 for proper evaluation.

**Files:**
- Results: `research/benchmark_finetuned_results.json`
- Model: `models/wildlife_yolov8n_best.pt`
- Training script: `research/fine_tune.py`
- Training data creation: `research/create_training_data.py`
- Training info: `models/wildlife_yolov8n_info.json`

---

## Experiment 4: Proper Train/Test Split (Data Leakage Fix)

**Date:** 2026-04-28

**Model:** `wildlife_detector_best.pt` (6.2 MB, 3M params)

**Approach:** Fine-tune YOLOv8n with proper clip-level train/test split - entire clips held out for testing, never seen during training.

**Training Configuration:**
```python
base_model = "yolov8n.pt"
epochs = 30
batch_size = 8
image_size = 640
train_clips = 26  # 2-3 per species
test_clips = 12   # 1 per species (held out)
seed = 42
```

**Key Change from Experiment 3:**
- Split at **clip level**, not frame level
- 1 entire clip per species held out for testing
- No frames from test clips appear in training data
- Proper measurement of generalization ability

**Results:**

**Test Images (extracted from held-out clips):**
- Test images: 160
- Ground truth boxes: 211
- Predictions: 131
- **Frame-level species accuracy: 65.6%**

**Clip-Level Evaluation (on original test videos):**
- Test clips: 12 (1 per species)
- Clips correctly classified: 6
- **Clip-level accuracy: 50.0%**

**Per-Species Results:**

| Species | Result | Prediction | Analysis |
|---------|--------|------------|----------|
| Bear | ✓ 100% | Correct | Large, distinctive |
| Elk | ✓ 100% | Correct | Large, distinctive |
| Fox | ✓ 100% | Correct | Enough training data |
| Horse | ✓ 100% | Correct | Large, distinctive |
| Moose | ✓ 100% | Correct | Large, distinctive |
| Wild Boar | ✓ 100% | Correct | Distinctive shape |
| Coyote | ✗ 0% | → fox | Visually similar to fox |
| Deer | ✗ 0% | → moose | Visually similar to moose |
| Goat | ✗ 0% | → fox | Limited training data |
| Opossum | ✗ 0% | → bear | Limited training data |
| Raccoon | ✗ 0% | No detection | Small, insufficient data |
| Skunk | ✗ 0% | No detection | Small, insufficient data |

**Training Validation Metrics (30 epochs):**
- mAP50: 0.397
- Best classes: Bear (0.995), Fox (0.783), Elk (0.656)
- Worst classes: Coyote (0.022), Raccoon (0.004), Skunk (0.006)

**Comparison to Experiment 3 (Data Leakage):**

| Metric | Exp 3 (Leaky) | Exp 4 (Proper) |
|--------|---------------|----------------|
| Claimed Accuracy | 98.5% | 50% |
| Test Data | Same as train | Held-out clips |
| Raccoon | 69% | 0% |
| Deer | 99.8% | 0% |
| Valid Measurement | ❌ No | ✓ Yes |

**Conclusion:**
- True generalization accuracy is **50%**, not 98.5%
- Large, distinctive animals (bear, elk, moose, horse) detected well
- Small animals (raccoon, skunk) need significantly more training data
- Visually similar species (coyote/fox, deer/moose) are confused
- Current dataset (26 train clips) is insufficient for reliable species ID
- **Next step: Path C - Gather more training data for underrepresented species**

**Files:**
- Results: `research/evaluation_results.json`
- Model: `models/wildlife_detector_best.pt`
- Training script: `research/train.py`
- Dataset setup: `research/setup_dataset.py`
- Evaluation script: `research/evaluate.py`
- Split info: `research/datasets/splits.json`

---

## Experiment Template

Copy this for new experiments:

```markdown
## Experiment N: [Title]

**Date:** YYYY-MM-DD

**Model:** [model file or path]

**Approach:** [brief description]

**Configuration:**
[code block with settings]

**Changes from Previous:**
- [list changes]

**Results:**
- **Total detections:** N
- **Average FPS:** X.X
- **Key metrics:** ...

**Species Accuracy:** [table or summary]

**Conclusion:**
[what we learned, next steps]

**Files:**
- Results: [path]
- Model: [path]
- Training script: [path if applicable]
```

---

## Next Experiments to Try

1. ~~**MegaDetector baseline**~~ - DONE (Experiment 2)
2. **Two-stage pipeline** - MegaDetector (early warning) → Fine-tuned YOLO on crops
3. ~~**Fine-tune YOLOv8n**~~ - DONE (Experiment 3) - DATA LEAKAGE, invalid
4. ~~**Proper train/test split**~~ - DONE (Experiment 4) - 50% real accuracy
5. **Path C: Add more training data** - IN PROGRESS
   - Download ENA24-detection dataset (10K images, 23 species)
   - Focus on: raccoon, skunk, coyote, deer, goat, opossum
6. **Edge optimization** - ONNX/TensorRT export for faster inference
7. **Night vision testing** - Test on infrared/low-light footage

## Current Recommendation

**⚠️ Current model is NOT production-ready** (50% accuracy)

**Immediate Priority: Path C - More Training Data**
- Download external datasets (ENA24, iNaturalist)
- Focus on underperforming species
- Re-train and re-evaluate

**For MVP (animal warning only):**
- MegaDetector alone is sufficient (95% detection rate)
- Does not identify species, just "animal detected"

**For species ID (future):**
- Need significantly more training data
- Consider two-stage pipeline for reliability
