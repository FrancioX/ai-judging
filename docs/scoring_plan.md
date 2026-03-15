# Plan: ST-GCN Score Regressor for Snowboard Freeride Runs

## TL;DR

Build a lightweight ST-GCN + MLP regressor that predicts a freeride competition score from a 3D pose sequence (H36M-17 skeleton). Start with **Snowboard Men only** (22 usable videos, scores 8.67–80). Evaluate with Leave-One-Out CV using Spearman's ρ and relative L2 distance. Include a hand-crafted feature baseline for comparison.

> **Why Snowboard Men only?** Ski and snowboard have fundamentally different biomechanics (edge-to-edge vs parallel stance, different rotation mechanics). Men and women are scored on different scales. Mixing categories would add noise. If the model shows signal, train **separate models per category** rather than a single cross-category model.

## Current State

- **3D poses**: H36M-17 (17 joints × 3 coords), normalized person-centered coordinates, output by MotionBERT
- **Only 3 of 22** snowboard videos have 3D poses computed → recompute all videos
- **No existing scoring module** in `src/`

## Data Inventory — Snowboard Men

| # | Video Stem | Score | 3D Poses |
|---|-----------|-------|:--------:|
| 1 | Snowboard Men_1_80_Fabian Knözinger | 80.00 | ✓ |
| 2 | Snowboard Men_2_71.67_Aleksandr Girnik | 71.67 | ✗ |
| 3 | Snowboard Men_3_69.33_Jakob Weger | 69.33 | ✗ |
| 4 | Snowboard Men_4_67.33_Pol Sabidó Juvé | 67.33 | ✗ |
| 5 | Snowboard Men_5_64_Tom Leclercq Chabenat | 64.00 | ✗ |
| 6 | Snowboard Men_6_63_Tristan Legru-Yildiz | 63.00 | ✗ |
| 7 | Snowboard Men_7_61.67_Soini Turunen | 61.67 | ✗ |
| 8 | Snowboard Men_8_56_Lucas Muñoz | 56.00 | ✗ |
| 9 | Snowboard Men_9_55_Matthew Podelnyk | 55.00 | ✗ |
| 10 | Snowboard Men_10_54.33_Cedric Giraudeau | 54.33 | ✗ |
| 11 | Snowboard Men_11_53_Nicolas Lagger | 53.00 | ✗ |
| 12 | Snowboard Men_12_52.33_Adriano Cardillo | 52.33 | ✗ |
| 13 | Snowboard Men_13_51.67_Alexandre Fournier-Simu | 51.67 | ✗ |
| 14 | Snowboard Men_14_48.67_Taketo Kinoshita | 48.67 | ✗ |
| 15 | Snowboard Men_15_45_Theodor Salen | 45.00 | ✗ |
| 16 | Snowboard Men_16_40_Jonatan Laland | 40.00 | ✓ |
| 17 | Snowboard Men_17_35_Quentin Puydenus | 35.00 | ✓ |
| 18 | Snowboard Men_18_31_Tom Hill | 31.00 | ✗ |
| 19 | Snowboard Men_19_19_Thomas Leisi | 19.00 | ✗ |
| 20 | Snowboard Men_20_15_Oscar Weatherall | 15.00 | ✗ |
| 21 | Snowboard Men_21_12_Daniel Nufer | 12.00 | ✗ |
| 22 | Snowboard Men_22_8.67_Gregory Whitehead | 8.67 | ✗ |

Excluded: `Snowboard Men_23_0_Martxelo Urruzola` (score = 0).

Score range: **8.67 – 80.00**, median ≈ 54.

---

## Phase 1 — Data Preparation

### Step 1: Batch-process all Snowboard Men videos

Rerun the full pipeline (frames → segmentation → tracking → pose_2d → pose_3d) for all 22 usable Snowboard Men videos (excluding only the 0-score video) so every sample is generated with a consistent configuration and code version.

Script: `scripts/batch_snowboard_pipeline.py` — loops over all usable Snowboard Men MP4s and calls `src.pipeline` for each (force full recomputation).

### Step 2: Create the scoring dataset module

`src/scoring/dataset.py` — PyTorch Dataset that:
- Scans `output/poses_3d/` for `Snowboard Men_*` directories
- Parses score from directory name (second number field)
- Loads `poses_3d.json` → tensor `(T, 17, 3)`
- Variable-length handling: pad to max sequence length + boolean mask
- Data augmentation (critical for N=22):
  - Temporal random crop (subsequences)
  - Left-right mirror (swap L/R joint indices)
  - Gaussian noise on coordinates

---

## Phase 2 — Architecture

### Step 3: H36M-17 graph definition

`src/scoring/graph.py` — defines:
- Adjacency matrix A (17×17) for the H36M-17 skeleton (16 bones + self-loops)
- Centripetal spatial partitioning (root-ward / self / leaf-ward subsets)

H36M-17 skeleton (native pipeline output — no conversion needed):
```
pelvis(0) → right_hip(1) → right_knee(2) → right_ankle(3)
pelvis(0) → left_hip(4) → left_knee(5) → left_ankle(6)
pelvis(0) → spine(7) → thorax(8) → neck(9) → head(10)
thorax(8) → left_shoulder(11) → left_elbow(12) → left_wrist(13)
thorax(8) → right_shoulder(14) → right_elbow(15) → right_wrist(16)
```

### Step 4: ST-GCN regressor

`src/scoring/model.py` — lightweight from-scratch ST-GCN:

```
Input: (B, 3, T, 17)  — batch × xyz × time × joints

ST-GCN Block 1:  SpatialGraphConv(3→64)   + TemporalConv1d(k=9)            + BN + ReLU + Drop(0.2)
ST-GCN Block 2:  SpatialGraphConv(64→128)  + TemporalConv1d(k=9, stride=2)  + BN + ReLU + Drop(0.2)
ST-GCN Block 3:  SpatialGraphConv(128→256) + TemporalConv1d(k=9, stride=2)  + BN + ReLU + Drop(0.2)

Temporal Pooling:  mean + std over T → concat → (B, 512, 17)
Spatial Pooling:   mean over joints  → (B, 512)

MLP Head: Linear(512→128) → ReLU → Drop(0.3) → Linear(128→1)

Output: predicted score (scalar)
```

### Step 5: Hand-crafted feature baseline

`src/scoring/baseline.py`:
- Per-frame features: 12 joint angles (knees, hips, elbows, shoulders, spine), velocities, accelerations, body height ratio
- Temporal aggregation: mean, std, min, max, range, skewness → ~150-dim feature vector
- Regressor: Ridge regression (sklearn) — robust with small N

---

## Phase 3 — Training & Evaluation

### Step 6: Training loop with LOOCV

`src/scoring/train.py`:
- **Loss**: SmoothL1Loss (Huber) — robust to score outliers
- **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-2
- **Schedule**: CosineAnnealingLR over 300 epochs
- **Cross-validation**: Leave-One-Out (22 folds) — each fold trains on 21, predicts on 1
- **Early stopping**: patience=50 on validation loss (hold out 1 video from training set)
- Score normalization: min-max scale targets to [0, 1] during training, rescale predictions for metrics

### Step 7: Evaluation metrics

`src/scoring/evaluate.py`:
- **Spearman's rank correlation** (ρ): measures ranking quality — do predicted ranks match actual ranks?
- **Relative L2 distance**: `mean(|y_pred - y_true| / y_true)` — measures percentage error
- Report: per-fold predictions table, aggregate metrics, prediction vs actual scatter plot

### Step 8: Run experiments and compare

1. Train ST-GCN regressor with LOOCV → collect all 22 predictions → compute metrics
2. Train baseline Ridge regressor with LOOCV → collect all 22 predictions → compute metrics
3. Compare Spearman's ρ and relative L2 for both approaches
4. Analyze failure cases: which videos are hardest to rank correctly?

---

## File Structure

```
src/scoring/
  __init__.py       # empty
  dataset.py        # PyTorch Dataset for pose sequences
  graph.py          # H36M-17 adjacency matrix + partitioning
  model.py          # ST-GCN + MLP regressor
  baseline.py       # Hand-crafted features + Ridge regression
  train.py          # Training loop + LOOCV
  evaluate.py       # Metrics (Spearman ρ + relative L2)
scripts/
  batch_snowboard_pipeline.py  # batch processing
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **H36M-17 skeleton** | Native pipeline output — no conversion layer needed |
| **From-scratch ST-GCN** (no pretrained) | N=22 is too small to benefit from fine-tuning a large pretrained action-recognition backbone |
| **Leave-One-Out CV** | Only viable cross-validation strategy for N=22; k-fold would be too noisy |
| **Data augmentation** | Main regularization lever with 22 samples |
| **Normalized person-centered coords** | Use pipeline output directly — no reprojection needed |
| **Snowboard Men only** | Biomechanics differ across sport and gender; mixing would add noise |
| **Separate baseline** | Ridge on hand-crafted features tells us whether graph structure adds value beyond simple statistics |

## Verification Checkpoints

1. **After Step 1**: All 22 `output/poses_3d/Snowboard Men_*/poses_3d.json` files exist with reasonable frame counts
2. **After Step 2**: Unit test — dataset loads 22 videos, shapes are `(T, 17, 3)`, augmentations produce valid outputs
3. **After Step 4**: Smoke test — forward pass with random `(1, 3, 100, 17)` input → scalar output
4. **After Step 6**: LOOCV produces exactly 22 predictions, one per video
5. **After Step 7**: Spearman's ρ > 0 (better than random); baseline comparison table

## Further Considerations

1. **Sequence length as a feature**: Run duration may correlate with score (crashes = short runs = low scores). Consider feeding sequence length to the MLP head alongside the ST-GCN embedding.
2. **Score distribution**: Roughly uniform 8.67–80, but sparse at the low end (8–20).
3. **Future scaling**: If signal is detected, train separate models per category (Ski Men, Snowboard Women, etc.) rather than a cross-category model.
