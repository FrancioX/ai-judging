# Experiment Log — Tracking Improvements

All experiments measured against 3 annotated ground-truth videos. Baseline recorded before any changes.

---

## Baseline (pre-experiments)

**Date:** 2026-02-27

| Video | Mean Error (px) | HOTA |
|-------|---------------:|-----:|
| Arno Vuarnier | 63.5 | 0.732 |
| Andreas Bakke | 18.2 | 0.895 |
| Lach Powell | 36.9 | 0.880 |
| **Overall** | **39.5** | **0.836** |

---

## Experiment 1 — Velocity Consistency Scoring

**Date:** 2026-02-27
**Goal:** Add a velocity-consistency term to candidate scoring during merge conflict resolution. The skier's implied velocity (direction + magnitude) should match recent motion history, distinguishing the moving athlete from stationary bystanders.

**Implementation:** Added `_velocity_consistency()` helper that scores candidates using weighted-mean cosine similarity (65%) + magnitude consistency (35%) against a sliding window of recent velocities (`deque(maxlen=5)`). Integrated as signal #5 in the combined scoring formula.

**Files changed:** `src/tracking/tracker.py`, `config.yaml`, `src/pipeline.py`
**Tests added:** `tests/test_velocity_consistency.py` (15 tests)

### Iteration 1a — w_velocity=0.8, no single-candidate history update

| Video | Mean Error (px) | HOTA | Δ vs baseline |
|-------|---------------:|-----:|:--------------|
| Arno | 63.5 | 0.732 | — |
| Andreas | **81.1** | — | **Major regression** |
| Lach | 36.9 | 0.880 | — |

**Problem:** Velocity history only built during multi-candidate frames, so it was stale/empty when conflicts arose. Weight too high.

### Iteration 1b — w_velocity=0.4, single-candidate history update added

| Video | Mean Error (px) | HOTA | Δ vs baseline |
|-------|---------------:|-----:|:--------------|
| Arno | 63.5 | 0.732 | — |
| Andreas | **35.6** | — | **Regression** |
| Lach | 36.9 | 0.880 | — |

**Problem:** Still too aggressive during conflict amplification phase.

### Iteration 1c — w_velocity=0.4, removed conflict-time velocity boost

| Video | Mean Error (px) | HOTA | Δ vs baseline |
|-------|---------------:|-----:|:--------------|
| Arno | 63.5 | 0.732 | — |
| Andreas | 18.2 | 0.895 | — |
| Lach | 36.9 | 0.880 | — |
| **Overall** | **39.5** | **0.836** | **No change** |

**Conclusion:** At w=0.4, the velocity signal contributes ≤0.4 points to a ~7.0 combined score — too weak to flip any decisions. Higher weights cause regressions because velocity history gets "poisoned" when a wrong candidate is chosen. The signal is fundamentally redundant with OF agreement (which already encodes motion direction). **Kept at w=0.4** for the velocity data it provides in the manifest (useful for downstream aerial detection/scoring).

---

## Experiment 2 — Synthetic OF Candidates

**Date:** 2026-02-27
**Goal:** When no YOLO detection is near the OF-predicted skier position, inject a synthetic candidate at the OF position. This addresses the root cause of aerial-phase identity switches: when the skier is mid-air and undetected, the bystander is the ONLY candidate, so it wins by default.

**Implementation:** In `_score_pass()`, after the OF proximity filter rejects all candidates (none within `2 × of_tight_radius_px`), inject a synthetic bbox centered on the OF prediction with the previous bbox dimensions and `confidence=0.3`. OF agreement score is zeroed for synthetic candidates to prevent circular self-reinforcement.

**Files changed:** `src/tracking/tracker.py`, `config.yaml`, `src/pipeline.py`
**Config param:** `of_synthetic_confidence: 0.3`

### Iteration 2a — Initial implementation (inject when no candidate within of_tight_radius_px)

| Video | Mean Error (px) | HOTA | Δ vs baseline |
|-------|---------------:|-----:|:--------------|
| Arno | **498.8** | 0.279 | **Catastrophic regression** |
| Andreas | **141.0** | 0.572 | **Major regression** |
| Lach | **112.0** | 0.549 | **Major regression** |

**Problem:** Two bugs: (1) synthetic candidates got full OF agreement bonus (1.5 × 3.0 = 4.5 pts during conflicts) — tautologically perfect since they ARE the OF prediction; (2) injection threshold was too loose, competing with valid nearby detections.

### Iteration 2b — Fixed: inject only when near_of is empty + zero OF agreement for synthetics

| Video | Mean Error (px) | HOTA | Δ vs baseline |
|-------|---------------:|-----:|:--------------|
| Arno | **42.4** | **0.756** | **-33.1% error, +3.3% HOTA** |
| Andreas | 18.2 | 0.896 | No change |
| Lach | 36.9 | 0.880 | No change |
| **Overall** | **32.5** | **0.844** | **-17.7% error, +1.0% HOTA** |

**Conclusion:** Significant improvement on the hardest video (Arno) with no regression on others. The fix was to only inject synthetics when NO real detection passes the OF proximity filter (all candidates >300px from OF prediction), and to zero out the OF agreement score for synthetic candidates to avoid circular self-reinforcement.

---

## Experiment 3 — Camera-Motion-Compensated Velocity

**Date:** 2026-02-27
**Goal:** Subtract estimated camera motion (pan/tilt) from candidate velocities so bystanders → ~zero residual velocity and skier → large residual velocity. The CMC infrastructure (ORB feature matching + RANSAC homography) already existed in the gap-filling phase.

**Implementation:** Computed `_compute_camera_motion()` during `_score_pass()` for both single- and multi-candidate frames. Subtracted `(cam_dx, cam_dy)` from raw pixel displacements before passing to `_velocity_consistency()` and before appending to velocity history.

**Files changed:** `src/tracking/tracker.py`, `config.yaml`, `src/pipeline.py`

### Iteration 3a — CMC velocity, w_velocity=1.2

| Video | Mean Error (px) | HOTA | Δ vs Exp 2b |
|-------|---------------:|-----:|:------------|
| Arno | **97.0** | 0.696 | **Major regression** |
| Andreas | **69.4** | 0.771 | **Major regression** |
| Lach | 40.9 | 0.878 | Slight regression |
| **Overall** | **69.1** | **0.782** | **-112% error** |

### Iteration 3b — CMC velocity, w_velocity=0.6

| Video | Mean Error (px) | HOTA | Δ vs Exp 2b |
|-------|---------------:|-----:|:------------|
| Arno | 96.5 | 0.699 | **Major regression** |
| Andreas | 57.9 | 0.815 | **Major regression** |
| Lach | 36.9 | 0.880 | No change |
| **Overall** | **63.8** | **0.798** | **-96% error** |

### Iteration 3c — Raw pixel velocity (no CMC), w_velocity=0.6

| Video | Mean Error (px) | HOTA | Δ vs Exp 2b |
|-------|---------------:|-----:|:------------|
| Arno | 96.5 | 0.699 | **Major regression** |
| Andreas | 57.9 | 0.815 | **Major regression** |
| Lach | 36.9 | 0.880 | No change |
| **Overall** | **63.8** | **0.798** | **-96% error** |

**Key finding:** Iterations 3b and 3c produced identical results — the CMC subtraction had zero differentiating effect. The regression was caused entirely by increasing `w_velocity` from 0.4 to 0.6, not by CMC noise. The velocity signal suffers from a positive feedback loop: wrong picks enter the history and reinforce the wrong trajectory.

**Conclusion:** Reverted to w_velocity=0.4 (no CMC). Removed CMC parameters from `_resolve_merge_conflicts`. The velocity-consistency signal cannot self-bootstrap reliably at any weight above 0.4.

---

## Current Best — Summary

**Config:** `w_velocity=0.4`, `of_synthetic_confidence=0.3`, synthetic OF candidates active, OF agreement zeroed for synthetics.

| Video | Mean Error (px) | HOTA |
|-------|---------------:|-----:|
| Arno Vuarnier | 42.4 | 0.756 |
| Andreas Bakke | 18.2 | 0.896 |
| Lach Powell | 36.9 | 0.880 |
| **Overall** | **32.5** | **0.844** |

**Improvement vs original baseline:** -17.7% mean error, +1.0% HOTA.
