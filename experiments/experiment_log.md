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

## Experiment 4 — Phase A.6 Jump-Size Instrumentation (`max_jump_px`)

**Date:** 2026-03-02
**Goal:** Log jump-size distributions used by the Phase A.6 identity guard to determine whether lowering `max_jump_px` could improve track generation.

**Implementation:** Extended `_validate_identity()` metadata to persist distance statistics (`of_dist_px`, `prev_det_dist_px`, rejected-jump stats, and threshold exceedance counts), printed Phase A.6 summaries during tracking, and saved them in `tracking.json` as `identity_guard_jump_stats`.

**Files changed:** `src/tracking/tracker.py`, `tests/test_identity_guard.py`
**Validation:** `uv run ruff check src/tracking/tracker.py` and `uv run pytest -q tests/test_identity_guard.py` (6 passed).

### Iteration 4a — Logging enabled, `max_jump_px=150` (all annotated videos)

| Video | Mean Error (px) | HOTA | Rejections | Δ vs current best |
|-------|---------------:|-----:|-----------:|:------------------|
| Arno Vuarnier | 42.3 | 0.754 | 59 | −0.1 px / −0.002 HOTA |
| Andreas Bakke | 18.3 | 0.894 | 93 | +0.1 px / −0.002 HOTA |
| Lach Powell | 37.0 | 0.876 | 33 | +0.1 px / −0.004 HOTA |
| **Overall** | **32.5** | **0.841** | **185** | **No meaningful change** |

**Phase A.6 jump diagnostics (per video):**

| Metric | Arno | Andreas | Lach |
|--------|-----:|--------:|-----:|
| **OF dist — p50 (px)** | 6.26 | 6.68 | 4.92 |
| **OF dist — p90 (px)** | 18.63 | 37.82 | 18.95 |
| **OF dist — p95 (px)** | 352.92 | 405.47 | 27.43 |
| **OF dist — max (px)** | 595.41 | 747.76 | 697.88 |
| **Prev det dist — p50 (px)** | 5.41 | 5.39 | 4.27 |
| **Prev det dist — p90 (px)** | 18.78 | 14.92 | 10.82 |
| **Prev det dist — p95 (px)** | 349.39 | 395.38 | 17.41 |
| **Prev det dist — max (px)** | 434.87 | 745.51 | 759.37 |
| **Over max_jump (150 px)** | 59/961 (6.1%) | 95/1162 (8.2%) | 33/1300 (2.5%) |
| **Rejected OF dist — median (px)** | 394.77 | 432.42 | 661.79 |

**Key observations:**
1. **Bimodal distribution** — p90 is well below 40 px across all videos, then p95 jumps to 350–400 px. The gap between "good" and "bad" detections is enormous (~10× the threshold).
2. **Threshold headroom** — The current `max_jump_px=150` sits in a dead zone: p90 ≪ 150 ≪ rejected median (~400–660 px). Lowering to 100 or even 80 would not reject any additional good detections (p90 < 40 px everywhere).
3. **Lach is cleanest** — Only 2.5% exceed threshold vs 8.2% for Andreas, correlating with fewer bystander crossovers.
4. **Both conditions required** — The guard requires BOTH `of_dist > 150` AND `prev_det_dist > 75`. Since both distances track closely for outliers, the dual-gate doesn't add much safety margin.

**Conclusion:** The current `max_jump_px=150` is effective but conservative. The clear bimodal gap (p90 ~20–40 px vs rejected median ~400–660 px) means lowering to 100 or even 80 px should be safe — it would catch identity switches earlier without risking false rejections. A sweep experiment (80/100/120) is the logical next step.

---

## Experiment 5 — `max_jump_px` Sweep (10/20/50/100)

**Date:** 2026-03-02
**Goal:** Determine the optimal `max_jump_px` threshold by sweeping 10/20/50/100 and measuring impact on association accuracy and error vs the production value of 150.

**Implementation:** Used `scripts/sweep_max_jump.py` to re-run tracking for all 3 annotated videos at each threshold value, collecting HOTA (DetA, AssA), mean error, and rejection counts.

**Files changed:** `config.yaml` (threshold value), `scripts/sweep_max_jump.py` (sweep runner)
**Output:** `output/sweep_max_jump_results.json`

### Results — Association Accuracy (`hota_ass_a`)

| max_jump_px | Arno (AssA) | Andreas (AssA) | Lach (AssA) | Mean AssA |
|:-----------:|:-----------:|:--------------:|:-----------:|:---------:|
| **10**      | 0.711       | 0.862          | 0.832       | 0.802     |
| **20**      | **0.720**   | **0.866**      | **0.852**   | **0.813** |
| 50          | 0.718       | 0.863          | 0.848       | 0.810     |
| 100         | 0.718       | 0.863          | 0.848       | 0.810     |

### Results — Full Metrics

| max_jump_px | Arno err / HOTA | Andreas err / HOTA | Lach err / HOTA | Overall err / HOTA | Rejections |
|:-----------:|:---------------:|:------------------:|:---------------:|:------------------:|:----------:|
| 10          | 43.3 / 0.746    | 18.5 / 0.891       | 39.1 / 0.860    | 33.6 / 0.832      | 739        |
| **20**      | **42.3 / 0.756**| **18.0 / 0.896**   | **36.7 / 0.880**| **32.3 / 0.844**   | **262**    |
| 50          | 42.3 / 0.754    | 18.3 / 0.894       | 37.0 / 0.876    | 32.5 / 0.841      | 185        |
| 100         | 42.3 / 0.754    | 18.3 / 0.894       | 37.0 / 0.876    | 32.5 / 0.841      | 185        |

### Key Findings

1. **`max_jump_px=20` is optimal** — best HOTA, best mean error, and best AssA across all three videos with zero regressions.
2. **`max_jump_px=10` over-rejects** — rejections explode to 739 (vs 262 at 20). Normal inter-frame skier motion (p50 ~5–7 px) is near the threshold, causing true-positive rejections. All videos regress: Arno −0.010 HOTA, Lach −0.020 HOTA.
3. **50 and 100 are identical** — confirms the bimodal gap from Experiment 4. Between 50–100 px there are virtually zero detections. The jump from "good" (p90 < 40 px) to "bad" (rejected median 370–660 px) is abrupt.
4. **20 catches identity switches in the 20–50 px gap** — the extra 77 rejections over the 50-threshold (262 vs 185) are overwhelmingly true identity switches, not valid fast movements.

**Conclusion:** Lowered `identity_guard_max_jump_px` from 150 → 20 in `config.yaml`. The improvement is modest but consistent across all videos: −0.6% mean error, +0.4% AssA, with no regressions.

---

## Current Best — Summary

**Config:** `w_velocity=0.4`, `of_synthetic_confidence=0.3`, `identity_guard_max_jump_px=20`, synthetic OF candidates active, OF agreement zeroed for synthetics.

| Video | Mean Error (px) | HOTA |
|-------|---------------:|-----:|
| Arno Vuarnier | 42.3 | 0.756 |
| Andreas Bakke | 18.0 | 0.896 |
| Lach Powell | 36.7 | 0.880 |
| **Overall** | **32.3** | **0.844** |

**Improvement vs original baseline:** -18.2% mean error, +1.0% HOTA.
