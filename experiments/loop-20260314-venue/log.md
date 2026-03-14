# Experiment Loop Log — Venue Mapping — 2026-03-14

Baseline: 538.3 px mean error (LoFTR homography, 1 keyframe accepted, 6 inliers).
LOO CV baseline (before this loop): N/A — system was essentially broken.

---

## Exp 1 — Global homography from tracking-center correspondences (venue_mapping_v2.py)

**Date**: 2026-03-14
**Hypothesis**: Fitting a perspective transform from video tracking-center coordinates to GT venue coordinates will outperform LoFTR feature matching on snow scenes.
**Implementation**: `scripts/venue_mapping_v2.py` — collects 10 GT×tracking overlapping frames, fits a RANSAC homography from (cx,cy)→(vx,vy), projects all frames.

| Video | mean_err_px | Δ vs baseline |
|-------|-------------|---------------|
| Ski Men_2_89_Andreas Bakke | 143.2 | −395.1 |

**Conclusion**: Big improvement over LoFTR (143px vs 538px), but only 6/10 inliers at 20px threshold (training residual 66.8px), showing a global homography is insufficient. Camera motion through the run means (cx,cy)→(vx,vy) is not globally consistent. **Rejected as production approach — a better interpolation method needed.**

---

## Exp 2a — GT-anchored linear interpolation (venue_mapping_v3.py, linear)

**Date**: 2026-03-14
**Hypothesis**: Direct use of GT annotations as anchors + linear temporal interpolation between them should give near-zero error on the current evaluation (which itself uses linear interpolation for GT).
**Implementation**: `scripts/venue_mapping_v3.py --interp linear` — at GT frames use exact GT position; between GT frames linearly interpolate by frame index; outside GT range use homography fallback.

| Video | non-LOO mean_err_px | LOO mean_err_px | LOO P90 | LOO within50px |
|-------|---------------------|-----------------|---------|----------------|
| Ski Men_2_89_Andreas Bakke | 0.0 | 34.3 | 63.7 | 80.0% |

**Conclusion**: Non-LOO is trivially 0 (evaluator uses same linear interpolation). LOO gives honest 34.3px mean. **Accepted as production baseline.**

---

## Exp 2b — GT-anchored tracking-arc-length interpolation (venue_mapping_v3.py, tracking)

**Date**: 2026-03-14
**Hypothesis**: Using cumulative tracking arc-length instead of frame index as interpolation parameter will better capture athlete speed variation.
**Implementation**: `scripts/venue_mapping_v3.py --interp tracking`

| Video | non-LOO mean_err_px | LOO mean_err_px | LOO P90 | LOO within50px |
|-------|---------------------|-----------------|---------|----------------|
| Ski Men_2_89_Andreas Bakke | 4.3 | 36.2 | 67.5 | 66.7% |

**Conclusion**: Slightly worse than linear on LOO. **Catastrophically fails for frames outside tracking range (990+): 264–369px errors.** Tracking arc-length is not available for those frames. Rejected.

---

## Exp 3 — Sparsity test: how many GT annotations are needed?

**Date**: 2026-03-14
**Method**: `scripts/venue_sparsity_test.py` — test N evenly-spaced subsets of GT annotations.

| N annotations | linear non-LOO mean | within50px |
|---------------|---------------------|-----------|
| 1 | 210.0 | 8.7% |
| 2 | 145.4 | 15.0% |
| 3 | 50.6 | 47.4% |
| 5 | 27.3 | 100.0% |
| 7 | 11.5 | 100.0% |
| 10 | 10.4 | 100.0% |
| 15 | 0.0 | 100.0% |

**Conclusion**: **5 evenly-spaced annotations per video suffices for 100% within 50px.** 7 annotations gives 11.5px mean. Informs recommended annotation effort.

---

## Exp 4 — Leave-one-out cross-validation (honest accuracy benchmark)

**Date**: 2026-03-14
**Method**: `scripts/venue_loo_eval.py` — hold out each GT frame in turn, predict, measure vs actual GT.

Original LOO results (with homography fallback for extrapolation):
- linear: 34.3px mean, 63.7px P90, 80% within 50px
- pchip: 33.0px mean, 57px P90, 80% within 50px

Worst-performing frames:
- Frame 150 (first GT): 68.3px — homography extrapolation backward is poor
- Frame 1370 (last GT): 92.6px — homography extrapolation forward is poor
- Frame 1250: 66.2px (linear) — non-linear athlete path, gap 200 frames

**Conclusion**: Homography fallback for extrapolation is the main failure mode. Linear trend extrapolation from nearest GT anchors would be much better (calculated 28px for frame 150, 60px for 1370).

---

## Exp 5 — Interpolation method comparison (venue_interp_test.py)

**Date**: 2026-03-14
**Methods tested**: linear, PCHIP, cubic spline, tracking arc-length (LOO CV).

| Method | LOO mean | LOO P90 | within50px |
|--------|----------|---------|-----------|
| linear | 34.3px | 63.7px | 80.0% |
| pchip | 33.0px | 57.0px | 80.0% |
| cubic | 36.1px | 57.9px | 80.0% |
| tracking | 103.8px | 314.4px | 46.7% |

**Conclusion**: PCHIP slightly better than linear (33.0 vs 34.3px); cubic is unstable (oscillation on some frames). Tracking mode catastrophically fails outside tracking range. PCHIP adopted.

---

## Exp 6 — PCHIP + linear trend extrapolation (venue_mapping_v3.py v2)

**Date**: 2026-03-14
**Hypothesis**: Replacing the homography fallback for out-of-range frames with linear trend extrapolation from the two nearest GT anchors will improve endpoint errors significantly.
**Implementation**: Updated `_build_dense_venue_positions` in `venue_mapping_v3.py`:
  - Removed `_fit_homography_fallback` call
  - For `fid < gt_first`: extrapolate backward from `kfs[0]`→`kfs[1]` slope
  - For `fid > gt_last`: extrapolate forward from `kfs[-2]`→`kfs[-1]` slope
  - Changed default `interp_mode` from `"linear"` to `"pchip"`

| Video | non-LOO mean_err_px | LOO mean_err_px | LOO P90 | LOO within50px |
|-------|---------------------|-----------------|---------|----------------|
| Ski Men_2_89_Andreas Bakke | 3.4 | 30.3 | 48.9 | 86.7% |

**Δ vs Exp 2a (previous best)**: LOO mean −4.0px, LOO P90 −14.8px, within50px +6.7pp

Worst frame improvements:
- Frame 150: 68.3px → 28.3px (linear trend extrapolation)
- Frame 1250: 66.2px → 43.7px (PCHIP handles non-linearity better)
- Frame 1350: 50.0px → 33.0px (PCHIP)

Remaining failures:
- Frame 550: 52.4px (PCHIP slight overshoot, linear gives 49.1px — both near boundary)
- Frame 1370: 60.0px (last GT frame, extrapolation limit)

**Accepted as new best.**

---

## Exp 7 — PCHIP/linear blend (alpha sweep)

**Date**: 2026-03-14
**Hypothesis**: A convex blend of PCHIP and linear predictions (blended_pred = α·PCHIP + (1-α)·linear) might fix the PCHIP overshoot at frames 750/1150 while keeping PCHIP's advantage at 1250/1350.
**Implementation**: Computed optimal α over range [0, 1] in 5% steps.

| alpha | LOO mean | LOO P90 | within50px |
|-------|----------|---------|-----------|
| 0.0 (linear) | 31.6 | 56.0 | 86.7% |
| 0.5 | ~30.6 | ~52 | 80.0% |
| 1.0 (PCHIP) | 30.3 | 48.9 | 86.7% |

**Conclusion**: Optimal α=1.0 (pure PCHIP). No blend improves over pure PCHIP. **Rejected.**

---

## Exp 8 — Per-segment similarity transform (tracking-guided)

**Date**: 2026-03-14
**Hypothesis**: Fit a 2D similarity transform (scale+rotation+translation) within each GT segment using tracking displacement → venue displacement, to better capture intra-segment path curvature.
**Implementation**: Within each GT segment [lo,hi], solve [a,-b;b,a]*[track_dx;track_dy]=[venue_dx;venue_dy] and apply to intermediate frame displacements.

| Method | LOO mean | LOO P90 | within50px |
|--------|----------|---------|-----------|
| PCHIP (current best) | 30.3 | 48.9 | 86.7% |
| Similarity transform | 521.8 | 999 | 6.7% |

**Conclusion**: Completely fails. Issues: (1) tracking-to-venue relationship is not consistent within a segment (camera pans); (2) for GT frames outside tracking range (990+), no tracking data available → 999px fallback; (3) tracking displacement is a poor proxy for venue displacement at 100-frame scale. **Rejected.**

---

## Final Best

**Method**: PCHIP interpolation + linear-trend extrapolation (`src/venue/venue_mapping.py`)

| Metric | Score |
|--------|-------|
| Standard eval (single video) | 3.4 px mean, 100% within 50px |
| LOO cross-validation | 30.3 px mean, 86.7% within 50px |
| vs. LoFTR baseline | −508 px (−94%) |

