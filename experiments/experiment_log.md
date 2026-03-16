# Experiment Log

Manual experiments outside of experiment loops. Append entries chronologically.

---

## ST-GCN Score Regressor — Initial Run

**Date:** 2026-03-16
**Branch:** `stgcn`
**Goal:** Test whether 3D pose sequences alone contain enough signal to predict freeride competition scores.

### Setup
- 22 Snowboard Men videos (scores 8.67–80.0), full pipeline through `poses_3d`
- Model: lightweight ST-GCN (3→64→128→256 channels) + MLP head, H36M-17 skeleton
- Baseline: hand-crafted 144-dim features (12 joint angles × 6 stats + 17 joint velocities × 4 stats + body height × 4 stats) → Ridge regression
- Evaluation: 22-fold LOOCV, Spearman ρ + relative L2 + MAE

### Results

| Metric | ST-GCN | Ridge Baseline |
|--------|--------|---------------|
| Spearman ρ | +0.119 | -0.248 |
| Relative L2 | 0.708 | 0.915 |
| MAE | 34.02 | 28.94 |

**Per-fold predictions (ST-GCN):**

| Athlete | True | Pred | \|Err\| |
|---------|------|------|--------|
| Fabian Knözinger | 80.00 | 35.79 | 44.21 |
| Aleksandr Girnik | 71.67 | 19.50 | 52.17 |
| Jakob Weger | 69.33 | 8.40 | 60.93 |
| Pol Sabidó Juvé | 67.33 | 16.75 | 50.58 |
| Tom Leclercq Chabenat | 64.00 | 16.10 | 47.90 |
| Tristan Legru-Yildiz | 63.00 | 21.21 | 41.79 |
| Soini Turunen | 61.67 | 15.10 | 46.57 |
| Lucas Muñoz | 56.00 | 17.86 | 38.14 |
| Matthew Podelnyk | 55.00 | 10.66 | 44.34 |
| Cedric Giraudeau | 54.33 | 13.58 | 40.75 |
| Nicolas Lagger | 53.00 | 14.54 | 38.46 |
| Adriano Cardillo | 52.33 | 12.30 | 40.03 |
| Alexandre Fournier-Simu | 51.67 | 11.11 | 40.56 |
| Taketo Kinoshita | 48.67 | 8.81 | 39.86 |
| Theodor Salen | 45.00 | 16.96 | 28.04 |
| Jonatan Laland | 40.00 | 12.94 | 27.06 |
| Quentin Puydenus | 35.00 | 16.49 | 18.51 |
| Tom Hill | 31.00 | 15.14 | 15.86 |
| Thomas Leisi | 19.00 | 9.38 | 9.62 |
| Oscar Weatherall | 15.00 | 13.20 | 1.80 |
| Daniel Nufer | 12.00 | 20.16 | 8.16 |
| Gregory Whitehead | 8.67 | 21.76 | 13.09 |

### Analysis / Failure Mode

**The model collapses to predicting 8–22 for all athletes regardless of true score.**

Root causes:

1. **Low pose variance across athletes.** Freeride runs consist mostly of turning/carving, where all athletes adopt geometrically similar stances. The ST-GCN input has minimal inter-athlete variance to regress against. Only during jumps (~10–20% of run time) do poses diverge meaningfully — but tricks are diluted in the full-sequence average pooling.

2. **Early stopping fires at epoch 51 in 21/22 folds.** The single-video validation set is too noisy: the model happens to predict the one val video accurately very early (val MAE as low as 0.11), then patience=50 locks in that checkpoint. The checkpoint reflects lucky initialisation, not learned representation.

3. **ST-GCN capacity/sample mismatch.** N=20 training examples for a network with millions of parameters; the model can memorise val but not generalise. The Ridge baseline has better MAE despite negative ρ.

### Conclusion

**3D pose alone is insufficient for this scoring task.** The discriminative signal lives in trajectory-level features (speed, fluidity, line choice) and aerial-phase-specific features (jump amplitude, rotation count, grab type) — not in full-run average pose statistics.

