# Experiment Log — Tracking Improvements

All experiments measured against 3 annotated ground-truth videos. Baseline recorded before any changes.

> **Annotation correction (2026-03-12):** GT annotations for 5 videos were found to be inaccurate and were re-annotated: Jordan Koch, Cedric Giraudeau, Loris Gonzalez, Coen Bennie-Faull, Quentin Puydenus. Experiments 8, 11, and 15 were re-run against corrected annotations on 2026-03-12. Other experiments' 18-video aggregates were not re-validated and should be treated as approximate. Snowboard videos are in `raw_videos_snowboard/`.

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

---

## Experiment 6 — No-OF Fast Iteration + Kalman Smoothing

**Date:** 2026-03-11
**Goal:** Validate a faster no-optical-flow iteration setup and test whether Kalman smoothing can recover part of the quality loss.

### Baseline (no OF, no identity guard)

**Config:** `optical_flow_method=none`, `identity_guard_enabled=false`, `smooth_window=0`.

| Metric | Value |
|-------|------:|
| Tracking runtime (18 videos, wall-clock) | **191.45 s** |
| Overall mean error (px) | **91.8** |
| Mean HOTA | **0.688** |

**Logs:** `output/experiments/exp_no_of_baseline_tracking_time.log`, `output/experiments/exp_no_of_baseline_eval.log`

### Iteration 6a — Enable Kalman smoothing

**Implementation:** Set `smooth_window=5` while keeping `optical_flow_method=none` and `identity_guard_enabled=false`.

**Files changed:** `config.yaml`

| Metric | No-OF baseline | Exp 6a (`smooth_window=5`) | Delta |
|-------|---------------:|---------------------------:|------:|
| Tracking runtime (s) | 191.45 | **189.25** | **-2.20 s** |
| Overall mean error (px) | 91.8 | **90.0** | **-1.8** |
| Mean HOTA | 0.688 | **0.684** | **-0.004** |

**Conclusion:** Smoothing gave a small error reduction and slightly faster runtime, but slightly reduced HOTA. It is neutral-to-slightly-positive for fast iteration quality.

---

## Experiment 7 — No-OF Previous-Detection Identity Guard

**Date:** 2026-03-11
**Goal:** Recover identity-switch robustness in no-OF mode using a lightweight guard based only on previous detection distance.

**Implementation:**
- Added a no-OF mode in `_validate_identity()` to reject detections when jump from previous accepted detection exceeds threshold.
- Wired Phase A.6 to run this guard when optical flow is disabled.
- Config: `identity_guard_enabled=true`, `identity_guard_max_jump_px=100`, `optical_flow_method=none`, `smooth_window=5`.

**Files changed:** `src/tracking/tracker.py`, `config.yaml`
**Validation:** `uv run pytest -q tests/test_identity_guard.py` (6 passed)

| Metric | Exp 6a (`smooth_window=5`) | Exp 7 (prev-det guard) | Delta |
|-------|---------------------------:|-----------------------:|------:|
| Tracking runtime (s) | 189.25 | **191.30** | **+2.05 s** |
| Overall mean error (px) | 90.0 | **198.9** | **+108.9** |
| Mean HOTA | 0.684 | **0.425** | **-0.259** |

**Conclusion:** Strong regression. The previous-detection-only guard over-rejects or locks onto wrong trajectories in no-OF mode and should not be used as-is.

---

## Experiment 8 — Raise `merge_score_threshold` to Reduce Candidate Noise

**Date:** 2026-03-11
**Goal:** Reduce wrong-person candidates in no-OF mode by merging fewer low-quality tracks.

**Implementation:** Raised `merge_score_threshold` from `0.3` → `0.5` while keeping the best no-OF baseline from Experiment 6 (`optical_flow_method=none`, `identity_guard_enabled=false`, `smooth_window=5`).

**Files changed:** `config.yaml`

| Video | Mean Error (px) | HOTA | Δ vs Exp 6a |
|-------|---------------:|-----:|:------------|
| Arno Vuarnier | **54.5** | **0.688** | **-36.8% error, +0.161 HOTA** |
| Andreas Bakke | **90.8** | **0.731** | **-0.5% error, +0.002 HOTA** |
| Lach Powell | **47.9** | **0.863** | **No material change** |
| **Overall (18 videos)** | **36.0** | **0.833** | *(re-validated 2026-03-12 with corrected annotations; previous value: 67.0px / 0.726)* |

| Runtime Metric | Exp 6a | Exp 8 | Delta |
|-------|--------:|------:|------:|
| Tracking runtime (s) | 189.25 | **179.59** | **-9.66 s** |

**Conclusion:** This is the best no-OF result so far. Reducing the merge pool at the track level sharply improved both runtime and quality. The no-OF tracker was being hurt primarily by low-scoring merged tracks injecting wrong candidates.

---

## Experiment 9 — Boost Track Stickiness & Continuity Weights

**Date:** 2026-03-11
**Goal:** Test whether stronger non-OF conflict-resolution weights can further improve no-OF tracking after cleaning the merge pool.

**Implementation:** Exposed `w_continuity` and `w_track_stickiness` as tracking config parameters, then set `w_continuity=1.2` and `w_track_stickiness=1.0` on top of the Experiment 8 configuration.

**Files changed:** `src/tracking/tracker.py`, `src/pipeline.py`, `config.yaml`
**Validation:** `uv run pytest -q tests/test_velocity_consistency.py tests/test_identity_guard.py` (21 passed)

| Video | Mean Error (px) | HOTA | Δ vs Exp 8 |
|-------|---------------:|-----:|:-----------|
| Arno Vuarnier | 54.5 | 0.688 | No change |
| Andreas Bakke | 90.8 | 0.731 | No change |
| Lach Powell | 47.9 | 0.863 | No change |
| **Overall (18 videos)** | **36.0** | **0.833** | **No change** *(18-video aggregate updated per annotation correction; Exp 9 not individually re-run)* |

| Runtime Metric | Exp 8 | Exp 9 | Delta |
|-------|------:|------:|------:|
| Tracking runtime (s) | 179.59 | **185.14** | **+5.55 s** |

**Conclusion:** After the merge pool was cleaned up by Experiment 8, increasing continuity and stickiness weights did not change any tracking decisions. The candidate set, not the scoring weights, was the dominant bottleneck.

---

## Experiment 10 — Confidence-Gated Merge Candidates

**Date:** 2026-03-11
**Goal:** Filter low-confidence detections out of the merge pool, not just low-scoring tracks.

**Implementation:** Added `merge_min_detection_conf` and skipped detections below this threshold when building the merged candidate pool. Ran with `merge_min_detection_conf=0.6` on top of the Experiment 9 configuration.

**Files changed:** `src/tracking/tracker.py`, `src/pipeline.py`, `config.yaml`
**Validation:** `uv run pytest -q tests/test_velocity_consistency.py tests/test_identity_guard.py` (21 passed)

| Video | Mean Error (px) | HOTA | Δ vs Exp 8 |
|-------|---------------:|-----:|:-----------|
| Arno Vuarnier | 54.6 | 0.677 | Slight regression |
| Andreas Bakke | 87.2 | 0.737 | Slight improvement |
| Lach Powell | 71.7 | 0.809 | Major regression |
| **Overall (18 videos)** | **69.5** | **0.715** | **+2.5 px, -0.011 HOTA** |

| Runtime Metric | Exp 8 | Exp 10 | Delta |
|-------|------:|-------:|------:|
| Tracking runtime (s) | 179.59 | **181.85** | **+2.26 s** |

**Conclusion:** Filtering detections by confidence was too aggressive. It helped some videos slightly, but it removed too many valid low-confidence skier detections on hard sequences, especially Lach, which increased interpolation burden and hurt association quality.

---

## Experiment 11 — Sweep `merge_score_threshold` (0.45–0.70)

**Date:** 2026-03-12
**Goal:** Verify whether `merge_score_threshold=0.5` is a true local optimum or if a nearby value improves no-OF tracking.

**Implementation:** Swept thresholds 0.45, 0.50, 0.55, 0.60, 0.65, 0.70 with fixed no-OF config: `optical_flow_method=none`, `identity_guard_enabled=false`, `smooth_window=5`, `merge_min_detection_conf=0.0`, `w_continuity=0.6`, `w_track_stickiness=0.4`.

**Files changed:** `config.yaml`, `scripts/sweep_merge_threshold.py`

### Sweep Results — 3 Annotated Videos

| Threshold | Arno err / HOTA | Andreas err / HOTA | Lach err / HOTA | Mean err | Mean HOTA |
|:---------:|:---------------:|:------------------:|:---------------:|:--------:|:---------:|
| 0.45 | 73.2 / 0.649 | 91.3 / 0.729 | 47.9 / 0.863 | 70.8 | 0.747 |
| 0.50 | 54.5 / 0.688 | 90.8 / 0.731 | 47.9 / 0.863 | 64.4 | 0.761 |
| 0.55 | 64.8 / 0.636 | 85.4 / 0.761 | 10.2 / 0.928 | 53.5 | 0.775 |
| **0.60** | **66.5 / 0.619** | **8.5 / 0.942** | **10.2 / 0.928** | **28.4** | **0.830** |
| 0.65 | 130.2 / 0.341 | 9.2 / 0.936 | 32.2 / 0.747 | 57.2 | 0.675 |
| 0.70 | 172.6 / 0.246 | 72.0 / 0.711 | 58.2 / 0.635 | 100.9 | 0.531 |

### Full 18-Video Evaluation at `merge_score_threshold=0.60`

*(Re-validated 2026-03-12 with corrected annotations)*

| Metric | Exp 8 (0.50) | Exp 11 (0.60) | Delta |
|--------|------------:|-------------:|------:|
| Overall mean error (px) | 36.0 | **24.9** | **-30.8%** |
| Mean HOTA | 0.833 | **0.855** | **+0.022** |

### Key Findings

1. **`0.60` is the clear optimum** — on the 3 annotated videos, mean error drops from 64.4 → 28.4 px and HOTA rises from 0.761 → 0.830. Andreas and Lach see dramatic improvements (90.8 → 8.5 px, 47.9 → 10.2 px) as the higher threshold cuts wrong-person tracks from the merge pool.
2. **Arno regresses slightly** (54.5 → 66.5 px) — the skier's own track fragments are near the threshold boundary, so some valid detections are lost. This is the main tradeoff.
3. **Beyond 0.60, quality collapses** — at 0.65, Arno's main tracks get excluded (130.2 px). At 0.70, all three videos degrade severely.
4. **Phase transition between 0.55 and 0.60** — Andreas jumps from 85.4 → 8.5 px, indicating that one specific bystander track scores between 0.55–0.60 and dominates when included.

**Conclusion:** Updated `merge_score_threshold` from 0.5 → 0.6 in config. This is the new best fixed-threshold no-OF configuration: **24.9 px mean error, 0.855 HOTA** (18 videos, corrected annotations).

---

## Current Best (no-OF) — Summary

**Config:** `merge_score_threshold=0.6`, `optical_flow_method=none`, `identity_guard_enabled=false`, `smooth_window=5`, `merge_min_detection_conf=0.0`, `w_continuity=0.6`, `w_track_stickiness=0.4`.

| Video | Mean Error (px) | HOTA |
|-------|---------------:|-----:|
| Arno Vuarnier | 66.5 | 0.619 |
| Andreas Bakke | 8.5 | 0.942 |
| Lach Powell | 10.2 | 0.928 |
| **Overall (18 videos)** | **24.9** | **0.855** | *(corrected annotations)* |

## Current Best (with OF) — Summary

**Config:** `w_velocity=0.4`, `of_synthetic_confidence=0.3`, `identity_guard_max_jump_px=20`, `merge_score_threshold=0.5`, synthetic OF candidates active, OF agreement zeroed for synthetics.

| Video | Mean Error (px) | HOTA |
|-------|---------------:|-----:|
| Arno Vuarnier | 42.3 | 0.756 |
| Andreas Bakke | 18.0 | 0.896 |
| Lach Powell | 36.7 | 0.880 |
| **Overall** | **32.3** | **0.844** |

---

## Experiment 12 — Reintroduce OF on Top of `merge_score_threshold=0.6`

**Date:** 2026-03-12
**Goal:** Test whether OF-based identity guard and gap-filling stack positively with the cleaner merge pool from Experiment 11.

**Implementation:** Swept OF configs with `merge_score_threshold=0.6`: identity guard at max_jump=50/100/150px. Also tested OF gap-fill only (no identity guard). Compared against the no-OF baseline and the production best (OF/jump=20/0.5).

**Files changed:** `scripts/sweep_of_jump.py`

### Results — 3 Annotated Videos

| Config | Arno err / HOTA | Andreas err / HOTA | Lach err / HOTA | Mean err | Mean HOTA |
|--------|:---------------:|:------------------:|:---------------:|:--------:|:---------:|
| **no-OF / 0.6** (Exp 11) | 66.5 / 0.619 | **8.5 / 0.942** | **10.2 / 0.928** | **28.4** | **0.830** |
| OF / jump=50 / 0.6 | 73.9 / 0.633 | 10.9 / 0.910 | 55.3 / 0.678 | 46.7 | 0.740 |
| OF / jump=100 / 0.6 | 73.9 / 0.633 | 10.9 / 0.909 | 55.3 / 0.678 | 46.7 | 0.740 |
| OF / jump=150 / 0.6 | 73.9 / 0.633 | 10.7 / 0.912 | 55.3 / 0.678 | 46.6 | 0.741 |
| *prod (OF/jump=20/0.5)* | *42.3 / 0.756* | *18.0 / 0.896* | *36.7 / 0.880* | *32.3* | *0.844* |

### Key Findings

1. **OF and the 0.6 threshold don't stack** — adding OF in any configuration makes every video worse than the no-OF/0.6 baseline. Lach suffers the most: 10.2 → 55.3 px.
2. **Identity guard jump threshold is irrelevant** — 50/100/150 produce nearly identical results, meaning the identity guard isn't the bottleneck. The OF gap-filling itself is the problem.
3. **OF changes the detected/interpolated ratio dramatically** — Lach goes from 1208/242 (no-OF) to 526/924 (OF at jump=20). The OF trace drifts onto bystanders during the gap-filling phase when the merge pool is small.
4. **Two distinct optima exist**: (a) OF/0.5 for best HOTA (0.844), (b) no-OF/0.6 for best 3-video mean error (28.4px) and best per-video scores on Andreas/Lach.

**Conclusion:** OF does not improve the 0.6 threshold config. The two optimisation paths (OF-based identity resolution vs merge-pool pruning) are complementary in principle but interfere in practice. Reverted config to no-OF/0.6 as current default.

---

---

## Experiment 13 — Candidate-Pool Diagnostics

**Date:** 2026-03-12
**Goal:** Instrument the tracker to expose merge-pool behavior per video, enabling evidence-driven threshold tuning.

**Implementation:** Enhanced `candidate_pool_stats` in the tracking manifest and stdout output with:
- `threshold_used`, `threshold_fallback` — which threshold was active and whether fallback was triggered
- `tracks_included` / `tracks_excluded` — full list of tracks with score and detection count
- `threshold_sensitivity_tracks` — tracks within ±0.05 of threshold (the "swing zone")
- `conflict_summary` — `conflict_frames`, `avg_score_margin`, `of_available_in_conflict` from the conflict scorer

`_resolve_merge_conflicts` now returns `(selected_obs, conflict_stats)`. Per-video summary printed at tracking time:
```
Candidate pool: 6 track(s) included, 7 excluded (threshold=0.6)
Near-boundary tracks: track 28 score=0.6443 (included), track 9 score=0.5808 (excluded)
Conflicts: 0 frame(s)
```

**Files changed:** `src/tracking/tracker.py`

| Metric | Before | After | Delta |
|--------|-------:|------:|------:|
| Overall mean error (px) | 24.9 | **24.9** | None *(values reflect corrected annotations)* |
| Mean HOTA | 0.855 | **0.855** | None |

**Conclusion:** Pure instrumentation — no tracking behavior changed, no regression. The new diagnostic fields in `tracking.json` and stdout will guide all future threshold and OF experiments.

---

---

## Experiment 14 — Diagnose Arno's Regression at `merge_score_threshold=0.6`

**Date:** 2026-03-12
**Goal:** Identify which tracks Arno loses at 0.6 vs 0.5 and whether they contain the actual skier.

**Implementation:** Created `scripts/diagnose_arno_threshold.py` — runs tracking at 0.50 and 0.60, extracts `candidate_pool_stats` diagnostics from each, computes median GT distance for every swing track using interpolated GT annotations.

**Files changed:** `scripts/diagnose_arno_threshold.py`

### Key Findings

| Track | Score@0.50 | n_dets | Median GT dist (px) | Frame range | Verdict |
|------:|:----------:|-------:|--------------------:|:-----------:|:--------|
| 7 | 0.517 | 18 | 182.8 | 47–72 | Ambiguous (early video, distant) |
| **70** | **0.588** | **10** | **85.4** | **1284–1293** | **Skier fragment** (closest to GT) |
| 77 | 0.547 | 27 | 217.7 | 1367–1400 | Likely bystander |

- **–55 detected frames** at threshold=0.6 vs 0.5 (all 55 come from these 3 swing tracks: 18+10+27=55).
- **Track 70 is the root cause**: 85.4px median GT distance in frames 1284–1293. This is a real skier fragment that 0.6 excludes. Tracks 7 and 77 are more distant and may be bystanders.
- The regression at 0.6 is not a broad fragmentation issue — it's a single 10-detection fragment of the skier near the end of the run.

**Conclusion:** A lightweight fix is possible — an adaptive threshold that detects near-GT swing tracks (e.g., by detection-count ratio or temporal non-overlap with the best track) could recover track 70 without re-admitting tracks 7 and 77.

---

---

## Experiment 15 — Hybrid Adaptive Threshold

**Date:** 2026-03-12
**Goal:** Auto-select between `merge_threshold_low=0.5` and `merge_threshold_high=0.6` per video to recover Arno without losing Andreas/Lach gains.

**Implementation:** Added `merge_threshold_adaptive` mode to `track_skier()`. Heuristic: compute `det_rate_high = sum(dets for tracks with score ≥ high_threshold) / n_frames`. If this rate is below `merge_threshold_min_overlap_ratio=0.55`, the skier's track is fragmented and fragments are being excluded → use low threshold. Otherwise, high threshold is correctly cutting bystanders.

Config: `merge_threshold_adaptive: true`, `merge_threshold_low: 0.5`, `merge_threshold_high: 0.6`, `merge_threshold_min_overlap_ratio: 0.55`

**Files changed:** `src/tracking/tracker.py`, `src/pipeline.py`, `config.yaml`

| Video | Adaptive selection | Mean Error (px) | HOTA | Δ vs Exp 11 (fixed 0.6) |
|-------|:------------------:|---------------:|-----:|:------------------------|
| Arno Vuarnier | 0.5 (det_rate=0.462 < 0.55) | **54.5** | **0.688** | **-12.0 px, +0.069 HOTA** |
| Andreas Bakke | 0.6 (det_rate=0.912 > 0.55) | **8.5** | **0.942** | No change |
| Lach Powell | 0.6 (det_rate=0.833 > 0.55) | **10.2** | **0.928** | No change |
| **Overall (18 videos)** | | **25.5** | **0.856** | **+0.6 px, +0.001 HOTA** *(re-validated 2026-03-12; previous: 55.4px / 0.739)* |

**Conclusion:** Adaptive threshold successfully recovers Arno's regression while preserving Andreas/Lach gains. The detection-rate heuristic is robust: Arno's fragmented track at 0.6 has det_rate=0.462 (well below 0.55), while high-quality sequences like Andreas (0.912) and Lach (0.833) stay at the higher threshold. With corrected annotations, Exp 11 (fixed 0.6) and Exp 15 (adaptive) are virtually tied on the 18-video aggregate (24.9 vs 25.5 px mean error, 0.855 vs 0.856 HOTA); adaptive remains preferred for primary videos due to Arno's recovery (-12px).

---

## Current Best (no-OF) — Updated Summary *(superseded by Exp 20)*

*(Re-validated 2026-03-12 with corrected annotations)*

**Config:** `merge_threshold_adaptive: true`, `merge_threshold_low: 0.5`, `merge_threshold_high: 0.6`, `merge_threshold_min_overlap_ratio: 0.55`, `optical_flow_method: none`, `identity_guard_enabled: false`, `smooth_window: 5`, `merge_min_detection_conf: 0.0`, `w_continuity: 0.6`, `w_track_stickiness: 0.4`.

| Video | Mean Error (px) | HOTA | Adaptive? |
|-------|---------------:|-----:|:----------:|
| Arno Vuarnier | 54.5 | 0.688 | → 0.5 |
| Andreas Bakke | 8.5 | 0.942 | → 0.6 |
| Lach Powell | 10.2 | 0.928 | → 0.6 |
| **Overall (18 videos)** | **25.5** | **0.856** | |

---

---

## Experiment 16 — Lightweight OF: Trace-Only (no gap-fill)

**Date:** 2026-03-12
**Goal:** Use OF trace for Phase A merge scoring/filtering without Phase B gap-fill. Hypothesis: trace gives merge benefit without gap-fill drift onto bystanders.

**Implementation:** Added `of_gap_fill_enabled` param to `track_skier()` (default `true`). When `false`, Phase B skips `_fill_gaps_optical_flow()` and falls back to `_fill_gaps()` (linear interpolation), while Phase A OF trace still runs. Tested with `optical_flow_method=auto`, `of_gap_fill_enabled=false`.

**Files changed:** `src/tracking/tracker.py`, `src/pipeline.py`, `config.yaml`

### Results — 3 Annotated Videos

| Video | Detected/Interp | Mean Error (px) | HOTA | Δ vs Exp 15 |
|-------|:---------------:|---------------:|-----:|:------------|
| Arno Vuarnier | 520/907 | 242.7 | 0.393 | **+188.2 px, -0.295 HOTA** |
| Andreas Bakke | 803/420 | 233.9 | 0.669 | **+225.4 px, -0.273 HOTA** |
| Lach Powell | 540/910 | 381.4 | 0.351 | **+371.2 px, -0.577 HOTA** |
| **Overall (18 videos)** | | **99.0** | **0.675** | **+43.6 px, -0.064 HOTA** |

**Root cause:** The Phase A OF trace in `_resolve_merge_conflicts._score_pass()` is NOT only scoring — it also filters candidates by proximity (keeps only dets within 2×radius=300px of trace, lines 766–779). When the OF trace drifts (because gap-fill is disabled and the trace can't re-anchor), legitimate skier detections get filtered out. Detected frames dropped from 714→520 (Arno), 1115→803 (Andreas), 1208→540 (Lach).

**Key finding:** Phase A trace and Phase B gap-fill are tightly coupled. The trace relies on gap-filling to stay on-track. Disabling gap-fill while keeping the trace active makes the trace drift and actively harm the candidate pool. The `of_gap_fill_enabled` parameter is committed for future experiments but trace-only mode is not viable without also disabling the Phase A candidate filter.

**Config reverted to:** `optical_flow_method: none`, `of_gap_fill_enabled: true`.

---

---

## Experiment 17 — Lightweight OF: Conditional Gap-Fill

**Date:** 2026-03-12
**Goal:** Use OF only for large gaps where linear interpolation is poor. Add drift guard to prevent OF-filled gaps from diverging too far from linear interpolation.

**Implementation:** Added two parameters to `_fill_gaps_optical_flow()`:
- `of_min_gap_for_fill=10`: gaps < 10 frames use linear interpolation; gaps ≥ 10 use OF
- `of_drift_guard_px=200.0`: if blended OF result > 200px from linear interpolation at same frame, fall back to linear

Tested with `optical_flow_method=auto`, `of_gap_fill_enabled=true`, `of_min_gap_for_fill=10`, `of_drift_guard_px=200.0`.

**Files changed:** `src/tracking/tracker.py`, `src/pipeline.py`, `config.yaml`

| Metric | Exp 15 (no-OF) | Exp 17 (conditional OF) | Delta |
|--------|---------------:|------------------------:|------:|
| Overall mean error (px) | 25.5 | **25.5** | None *(values reflect corrected annotations; Exp 17 not individually re-run)* |
| Mean HOTA | 0.856 | **0.856** | None |

**Conclusion:** Neutral result — conditional OF with drift guard matches no-OF exactly. The drift guard is working correctly (preventing OF drift from creating regressions), but the conditions where OF genuinely helps (large gaps + stable OF) are not common enough in these sequences to produce measurable gains. Config reverted to `optical_flow_method=none`. Infrastructure committed for future use.

---

---

## Experiment 18 — Lightweight OF: High-Confidence Re-Anchoring

**Date:** 2026-03-12
**Goal:** Prevent the Phase A OF trace from anchoring to low-confidence bystander detections by filtering candidates below a confidence threshold before re-anchoring.

**Implementation:** Added `of_reanchor_min_conf` parameter to `_build_of_trace()`. In `_find_closest_candidate`, candidates with `confidence < of_reanchor_min_conf` are skipped before the nearest-neighbor re-anchor search. Returns `None` if no confident candidate exists, keeping the trace extrapolating rather than snapping to a low-confidence detection.

Tested with `optical_flow_method=auto`, `of_reanchor_min_conf=0.7`, adaptive threshold active.

**Files changed:** `src/tracking/tracker.py`, `src/pipeline.py`, `config.yaml`

| Metric | Exp 15 (no-OF) | Exp 18 (of_reanchor_min_conf=0.7) | Delta |
|--------|---------------:|----------------------------------:|------:|
| Mean HOTA | 0.856 | **0.678** | **-0.178** *(Exp 15 baseline reflects corrected annotations; Exp 18 not re-run)* |

**Root cause:** Requiring confidence ≥ 0.7 for re-anchoring was too strict. Many valid skier detections have moderate confidence (0.5–0.7), especially on distant or partially-occluded frames. With re-anchoring disabled for these, the trace cannot follow the skier and instead drifts. The drifted trace then filters legitimate candidates via the Phase A proximity gate (2×150px), reducing detected frames across all three videos and increasing interpolation burden.

**Key finding:** High-conf re-anchoring backfires for the same architectural reason as Exp 16 — Phase A OF trace filtering and Phase B OF gap-fill are inseparable in the current design. Any perturbation that causes trace drift (disabling gap-fill, raising re-anchor threshold) cascades into candidate over-filtering. The only robust lightweight OF improvement would require decoupling Phase A scoring from Phase A filtering (i.e., use trace position as a soft score signal, not a hard proximity gate).

**Config reverted to:** `optical_flow_method: none`, `of_reanchor_min_conf: 0.0`.

---

## OF Architecture Analysis — Lightweight vs. Heavy Improvements

**Date:** 2026-03-12

### Summary: Why Lightweight OF Improvements Fail

Experiments 16, 17, and 18 all targeted specific aspects of OF without touching the core architecture. All three either regressed or were neutral. The root cause is a structural coupling in `tracker.py`:

1. **Phase A OF trace** (`_build_of_trace`) runs a forward Lucas-Kanade trace from the best anchor. Inside `_resolve_merge_conflicts._score_pass()`, this trace is used for **two distinct purposes**: (a) as a soft score signal (OF agreement term) and (b) as a **hard proximity gate** that discards candidates > 2×`identity_guard_max_jump_px` from the trace.
2. **Phase B OF gap-fill** (`_fill_gaps_optical_flow`) keeps the Phase A trace anchored across detection gaps by filling them in first. Without gap-fill, the trace drifts and the proximity gate misfires.

Any intervention that degrades trace quality — disabling gap-fill (Exp 16), restricting re-anchoring (Exp 18), or even just using conservative drift guards (Exp 17) — reduces detected frames via the proximity gate rather than improving quality. The OF machinery is coherent only when all parts run together.

### Lightweight Improvements That Could Work (Architectural Fixes)

The following targeted changes would decouple OF scoring from OF filtering without a full rewrite:

1. **Decouple trace scoring from trace filtering** (Medium effort, ~1 day): Change `_score_pass()` to use trace proximity as a *soft* score term (already partially done via `w_of_agreement`), and remove or relax the hard proximity gate. This would allow OF trace quality to contribute positively to merge decisions without gating out valid candidates when the trace drifts. Estimated impact: enables re-running Exps 16–18 safely.

2. **Per-gap OF decision with stable anchor check** (Medium effort, ~1 day): Before running OF on a gap, verify the forward and backward anchor detections are high-confidence (> 0.6) and close to GT-consistent positions. Only invoke OF if both anchors are reliable; otherwise use linear. This is a more principled version of Exp 17's `of_min_gap_for_fill` heuristic. Would prevent OF from amplifying errors when anchor quality is low.

3. **Phase A trace as scoring-only signal** (Small effort, ~2h): Add a flag `of_trace_filter_enabled: bool` (default `true`) to guard the proximity gate in `_score_pass()`. When `false`, trace runs only for the OF agreement score term — no candidates are filtered by proximity. This is the minimal change to test whether the trace *score signal* is beneficial independently of the *filter*.

### Heavy Improvements

1. **ReID Embeddings (OSNet-x0.25)** — Most practical heavy improvement. Trains a 128-dim appearance embedding per detection (MobileNet-class backbone, ~2–5 s/video on MPS). Embeddings replace or augment the spatial continuity term in conflict resolution, directly solving skier-vs-bystander confusion at the representation level rather than via position heuristics. New module `src/tracking/reid.py`. Requires `torchreid` dependency. Expected gain: strong improvement on sequences with bystanders at similar positions to the skier (Arno, Lach).

2. **Learned Optical Flow (RAFT)** — Replace Lucas-Kanade/Farnebäck with RAFT (Recurrent All-Pairs Field Transforms). RAFT is dramatically more accurate on textureless surfaces (snow, sky) and under large motion, both common in freeride skiing. Estimated runtime: 10–30 s/video on MPS (vs. <5 s for LK). Drop-in replacement for `_fill_gaps_optical_flow()` and `_build_of_trace()`. Biggest gain expected on sequences with large inter-frame motion (fast skiing, camera pans).

3. **Temporal Attention / Global Track Selection** — Replace the greedy forward merge scoring with a global transformer-based tracker that scores track assignments jointly over the entire video. Would solve fragmentation and identity switching at the source rather than patching them with heuristics. Heaviest option (~50–200 ms/video for a lightweight variant), likely overkill for 18-video competition judging. Consider only if ReID + improved OF cannot close the gap to the OF-based optimum (32.3 px / 0.844 HOTA).

**Recommended next step:** Implement the "Phase A trace as scoring-only signal" (item 3 above) as the minimal structural test, then add ReID as the primary heavy improvement.

---

---

## Experiment 19 — Structural Fix: Decouple Phase A Trace Filtering from Scoring

**Date:** 2026-03-12
**Goal:** Add `of_trace_filter_enabled` flag to guard the Phase A hard proximity gate so that trace can be used as a pure soft score signal without filtering candidates. This unblocks safe re-testing of Exps 16–18 variants.

**Implementation:** Added `of_trace_filter_enabled: bool = True` parameter to `_resolve_merge_conflicts()` and `track_skier()`. When `False`, the hard proximity gate (candidates > 2×`of_tight_radius_px` from trace discarded) is skipped entirely — trace still runs for the `w_of_agreement` score term. Wired through `src/pipeline.py` (both call sites) and `config.yaml`.

**Files changed:** `src/tracking/tracker.py`, `src/pipeline.py`, `config.yaml`
**Validation:** `uv run ruff check src/` ✓, `uv run pytest -q tests/test_velocity_consistency.py tests/test_identity_guard.py` (21 passed)

| Metric | Before | After | Delta |
|--------|-------:|------:|------:|
| Overall mean error (px) | 25.5 | **25.5** | None *(pure infrastructure)* |
| Mean HOTA | 0.856 | **0.856** | None |

**Conclusion:** Pure infrastructure change — no tracking behavior changed (default `of_trace_filter_enabled=true` preserves existing behavior exactly). The flag enables the next experiments: (a) testing `of_trace_filter_enabled=false` to measure whether the trace score signal is beneficial independently of the filter, and (b) safely re-running Exps 16–18 (lightweight OF variants) without the drift→filter cascade.

---

## Experiment 20 — OF Trace as Soft Score Signal (Filter Disabled)

**Date:** 2026-03-12
**Goal:** Test whether disabling the Phase A hard proximity gate (`of_trace_filter_enabled=false`) while keeping OF gap-fill active improves tracking vs the no-OF baseline (Exp 15). Enabled by the structural fix in Exp 19.

**Implementation:** Set `optical_flow_method=auto`, `of_trace_filter_enabled=false`, `of_gap_fill_enabled=true`, `identity_guard_enabled=false`, adaptive threshold active. Three configs tested on 3 annotated videos via `scripts/sweep_trace_filter.py`, then the winner run on all 18 via `--stop-after tracking`.

**Files changed:** `config.yaml`, `scripts/sweep_trace_filter.py`

### 3-Video Sweep Results

| Config | Arno err / HOTA | Andreas err / HOTA | Lach err / HOTA | Mean err | Mean HOTA | ms/frame (3-vid) |
|--------|:---------------:|:------------------:|:---------------:|:--------:|:---------:|:----------------:|
| no-OF baseline (Exp 15) | 54.5 / 0.688 | 8.5 / 0.942 | 10.2 / 0.928 | 24.4 | 0.853 | 6.33 |
| **OF trace score + gap-fill, no filter** | **51.4 / 0.734** | **8.7 / 0.932** | **9.2 / 0.931** | **23.1** | **0.866** | **48.0** |
| OF trace score only, no gap-fill | 54.5 / 0.688 | 8.5 / 0.942 | 10.2 / 0.928 | 24.4 | 0.853 | 33.6 |

**Key finding from sweep:** "Trace score only, no gap-fill" is **identical** to no-OF baseline — without gap-fill, the OF trace drifts and its soft score signal becomes noise. The gap-fill is what keeps the trace accurate and useful. "Trace score + gap-fill, no filter" wins because the trace contributes a useful soft scoring signal (via `w_of_agreement`) while the gap-fill keeps it on-target, and removing the hard filter prevents valid far-from-trace candidates from being discarded when the trace briefly drifts.

### Full 18-Video Evaluation

| Metric | Exp 15 (no-OF) | Exp 20 (soft trace) | Delta |
|--------|---------------:|--------------------:|------:|
| Overall mean error (px) | 25.5 | **23.5** | **-2.0 px (-7.8%)** |
| Mean HOTA | 0.856 | **0.869** | **+0.013** |
| ms/frame (18-video) | **6.05** | ~45.2 | **+39.2 ms (+6.5×)** |

| Video | Mean Error (px) | HOTA | Δ vs Exp 15 |
|-------|---------------:|-----:|:------------|
| Arno Vuarnier | 51.4 | 0.734 | -3.1 px, +0.046 HOTA |
| Andreas Bakke | 8.7 | 0.932 | +0.2 px, -0.010 HOTA |
| Lach Powell | 9.2 | 0.931 | -1.0 px, +0.003 HOTA |
| **Overall (18 videos)** | **23.5** | **0.869** | **-2.0 px, +0.013 HOTA** |

**Conclusion:** Removing the hard proximity gate while keeping OF gap-fill is a genuine improvement over no-OF baseline. The trace score signal is beneficial when the gate cannot accidentally cut valid candidates. Andreas has a small regression (+0.2 px, -0.010 HOTA) but this is outweighed by broad gains elsewhere. **New best: 23.5 px mean error, 0.869 HOTA.** Config promoted to production default.

---

## Current Best — Updated Summary

*(2026-03-12)*

**Config:** `optical_flow_method: auto`, `of_trace_filter_enabled: false`, `of_gap_fill_enabled: true`, `merge_threshold_adaptive: true`, `merge_threshold_low: 0.5`, `merge_threshold_high: 0.6`, `merge_threshold_min_overlap_ratio: 0.55`, `identity_guard_enabled: false`, `smooth_window: 5`, `w_continuity: 0.6`, `w_track_stickiness: 0.4`.

| Video | Mean Error (px) | HOTA |
|-------|---------------:|-----:|
| Arno Vuarnier | 51.4 | 0.734 |
| Andreas Bakke | 8.7 | 0.932 |
| Lach Powell | 9.2 | 0.931 |
| **Overall (18 videos)** | **23.5** | **0.869** |

**Improvement vs Exp 15 (previous best):** -2.0 px (-7.8%), +0.013 HOTA.
**Improvement vs original baseline:** -40.5% mean error, +4.0% HOTA.

---

## Experiment 21 — Conditional Gap-Fill and High-Conf Re-Anchoring (filter disabled)

**Date:** 2026-03-13
**Goal:** Re-test Exp 17 (conditional gap-fill) and Exp 18 (high-conf re-anchoring) variants with `of_trace_filter_enabled=false` to see if they stack with the Exp 20 improvement.

**Implementation:** All configs use Exp 20 base (`optical_flow_method=auto`, `of_trace_filter_enabled=false`, `of_gap_fill_enabled=true`, adaptive threshold). Four configs swept via `scripts/sweep_exp21.py` on 3 annotated videos. ms/frame measured from `n_frames` in tracking manifest vs wall-clock runtime.

**Files changed:** `scripts/sweep_exp21.py`, `config.yaml`

### 3-Video Sweep Results

| Config | Arno err | Andreas err | Lach err | Mean err | HOTA | ms/frame |
|--------|:--------:|:-----------:|:--------:|:--------:|:----:|:--------:|
| Exp20 baseline (soft trace) | 51.4 | 8.7 | 9.2 | 23.1 | 0.866 | 45.2 |
| **Exp21a: cond. gap-fill (min_gap=10, drift=200)** | **47.9** | **8.7** | **9.2** | **21.9** | **0.866** | **42.4** |
| Exp21b: high-conf reanchor (min_conf=0.7) | 51.4 | 8.7 | 9.2 | 23.1 | 0.866 | 45.4 |
| Exp21c: both combined | 47.9 | 8.7 | 9.2 | 21.9 | 0.866 | 42.7 |

*No-OF baseline (Exp 15) for reference: ~6.4 ms/frame (26s / 4090 frames)*

### Key Findings

1. **Exp21a wins**: Conditional gap-fill (`of_min_gap_for_fill=10`, `of_drift_guard_px=200`) improves Arno by 3.5 px (-6.8%) with no regression on Andreas/Lach — and is **2.8 ms/frame faster** than Exp20 because short gaps use cheap linear interpolation instead of OF.
2. **Exp21b is neutral**: High-conf re-anchoring (`of_reanchor_min_conf=0.7`) has zero effect here. With `of_trace_filter_enabled=false` there is no longer a proximity gate that drifted trace position can misfire — the primary failure mode of Exp 18 is gone, but so is the mechanism by which re-anchoring would help.
3. **Exp21c = Exp21a**: High-conf re-anchoring adds nothing on top of conditional gap-fill.
4. **Why conditional gap-fill helps**: Short gaps (< 10 frames) are better served by linear interpolation — OF on short gaps sometimes latches onto background motion. Restricting OF to gaps ≥ 10 frames keeps it focused on the cases where smooth interpolation fails (long aerial phases).

### Full 18-Video Evaluation — Exp21a

*(run after 3-video sweep confirmed improvement)*

| Metric | Exp 15 (no-OF) | Exp 20 | Exp 21a | Δ vs Exp 20 | Δ vs Exp 15 |
|--------|---------------:|-------:|--------:|:-----------:|:-----------:|
| Overall mean error (px) | 25.5 | 23.5 | **22.4** | **-1.1 px (-4.7%)** | **-3.1 px (-12.2%)** |
| Mean HOTA | 0.856 | 0.869 | **0.876** | **+0.007** | **+0.020** |
| ms/frame (18-video, measured) | **6.05** | ~45.2 | **41.9** | **-3.3 ms (-7.3%)** | +35.9 ms (+5.9×) |

**ms/frame source:** no-OF timed via `scripts/time_no_of_tracking.py` (temp dir, no output overwrite): 175.1s / 28,955 frames = 6.05 ms/frame. Exp 21a: 1,213.6s / 28,955 frames = 41.9 ms/frame.

| Video | Mean Error (px) | HOTA | ms/frame |
|-------|---------------:|-----:|:--------:|
| Arno Vuarnier | 47.9 | 0.734 | ~42.4 |
| Andreas Bakke | 8.7 | 0.932 | ~42.4 |
| Lach Powell | 9.2 | 0.931 | ~42.4 |
| Jordan Koch | 42.6 | 0.780 | 48.8 |
| Cedric Giraudeau | 13.1 | 0.906 | 42.8 |
| Emile Peizerat | 15.7 | 0.858 | 38.7 |
| Nicolas Lagger | 8.3 | 0.928 | 34.0 |
| Gabin Leonard | 58.5 | 0.867 | 33.0 |
| Adriano Cardillo | 9.8 | 0.913 | 41.6 |
| Maximilien Michel | 13.8 | 0.901 | 43.4 |
| Lucas Daines | 6.3 | 0.947 | 43.4 |
| Taketo Kinoshita | 12.5 | 0.950 | 35.5 |
| Tibo Mantero | 21.4 | 0.869 | 39.4 |
| Theodor Salen | 54.6 | 0.777 | 47.9 |
| Loris Gonzalez | 14.5 | 0.888 | 41.7 |
| Jonatan Laland | 4.6 | 0.955 | 42.1 |
| Coen Bennie-Faull | 14.0 | 0.889 | 40.1 |
| Quentin Puydenus | 47.8 | 0.753 | 51.1 |
| **Overall (18 videos)** | **22.4** | **0.876** | **~41.9** |

**Conclusion:** Conditional gap-fill stacks cleanly with the Exp 20 soft-trace improvement: -1.1 px, +0.007 HOTA, and 3.3 ms/frame faster. By restricting OF to gaps ≥ 10 frames, short gaps use cheap linear interpolation — eliminating short-gap OF errors while keeping OF for the long aerial phases where it matters. **New best: 22.4 px mean error, 0.876 HOTA, ~41.9 ms/frame.** Config promoted to production default.

---

## Current Best — Updated Summary

*(2026-03-13)*

### Best Overall — Exp 21a (OF, conditional gap-fill)

**Config:** `optical_flow_method: auto`, `of_trace_filter_enabled: false`, `of_gap_fill_enabled: true`, `of_min_gap_for_fill: 10`, `of_drift_guard_px: 200.0`, `merge_threshold_adaptive: true`, `merge_threshold_low: 0.5`, `merge_threshold_high: 0.6`, `merge_threshold_min_overlap_ratio: 0.55`, `identity_guard_enabled: false`, `smooth_window: 5`, `w_continuity: 0.6`, `w_track_stickiness: 0.4`.

| Video | Mean Error (px) | HOTA | ms/frame |
|-------|---------------:|-----:|---------:|
| Arno Vuarnier | 47.9 | 0.734 | ~42.4 |
| Andreas Bakke | 8.7 | 0.932 | ~42.4 |
| Lach Powell | 9.2 | 0.931 | ~42.4 |
| **Overall (18 videos)** | **22.4** | **0.876** | **41.9** |

**Improvement vs original baseline:** -43.3% mean error, +4.8% HOTA.

---

### Best Lightweight — Exp 15 (no-OF, adaptive threshold)

**Config:** `optical_flow_method: none`, `merge_threshold_adaptive: true`, `merge_threshold_low: 0.5`, `merge_threshold_high: 0.6`, `merge_threshold_min_overlap_ratio: 0.55`, `identity_guard_enabled: false`, `smooth_window: 5`, `w_continuity: 0.6`, `w_track_stickiness: 0.4`.

| Video | Mean Error (px) | HOTA | ms/frame |
|-------|---------------:|-----:|---------:|
| Arno Vuarnier | 54.5 | 0.688 | 6.21 |
| Andreas Bakke | 8.5 | 0.942 | 6.38 |
| Lach Powell | 10.2 | 0.928 | 6.09 |
| **Overall (18 videos)** | **25.5** | **0.856** | **6.05** |

**Use when:** real-time or near-real-time requirements; ~7× faster than Exp 21a at the cost of +3.1 px (+14%) mean error and -0.020 HOTA.

---

## Next Steps

1. **ReID embeddings** — implement `src/tracking/reid.py` with OSNet-x0.25 for appearance-based conflict resolution. Arno (47.9 px) and Quentin Puydenus (47.8 px) are the hardest videos — both likely need appearance-level discrimination.
