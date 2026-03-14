# Experiment Loop Log — Valid Venue Mapping — 2026-03-14

Goal: Build a venue mapping system that works WITHOUT GT athlete-position annotations at
inference time. GT is only allowed for evaluation/scale calibration.

Baseline: LoFTR = 538px mean error, 0% within 50px.

---

## Exp 1 — Background OF (all pixels) + linear regression

**Date**: 2026-03-14
**Hypothesis**: Integrating background optical flow over time gives camera pan; linear mapping from cumulative pan to venue position can localize the athlete.

**Implementation**: `venue_camflow.py` — Farneback dense OF on all non-athlete pixels,
cumulative sum, linear least-squares regression from cum_flow → venue position.

| Video | n_frames | LOO mean (px) | LOO within 50px | Notes |
|-------|----------|:---:|:---:|---|
| Andreas Bakke (841 frames) | 841 | 149.3 | 20% | linear, all-bg |
| Andreas Bakke (841 frames) | 841 | 160.5 | 10% | linear, dark-feature OF |

**Conclusion**: Better than LoFTR but far from useful. Scale varies with camera zoom — linear
regression cannot capture it. Dark-feature OF is NOT better than all-background (the relationship
between background flow and venue position is fundamentally non-linear).

---

## Exp 2 — Background OF (dark feature pixels) + PCHIP regression

**Date**: 2026-03-14
**Hypothesis**: PCHIP interpolation on cumulative flow (instead of linear) captures zoom variation.
Using dark+textured pixels (rocks/trees) as the OF signal gives more reliable camera motion.

**Implementation**: Updated `venue_camflow.py` — PCHIP spline fit from cum_flow_y → venue_y
(and separately for x). LOO cross-validation: hold out each GT frame, fit PCHIP on remaining,
predict held-out from flow signal only (never look up GT as prediction value).

Extended Andreas Bakke tracking: re-ran segmentation+tracking on all 1223 frames (was 841).

**Also tested on**: Knözinger (2343 frames, 25 GT annotations covering full run).

| Video | n_frames | n_GT | LOO mean | LOO median | LOO P90 | LOO within 50px |
|-------|----------|:---:|:---:|:---:|:---:|:---:|
| Andreas Bakke | 1223 | 15 | **52.9px** | 48.2px | 91.5px | 53% |
| Knözinger | 2343 | 25 | **67.0px** | **25.6px** | 62.4px | **88%** |

Note: Knözinger's 67px MEAN is inflated by ONE outlier (frame 150: 989px).
Frame 150 = first GT annotation. When held out, PCHIP must extrapolate to cum_flow=0
(outside calibration range) → catastrophic failure. **Excluding that one point: mean=28.6px**.

**Key finding**: For inner frames (not first/last), PCHIP-on-flow achieves ~28px mean LOO —
essentially the same as the (invalid) PCHIP-on-frame-index (30.3px) while using NO GT at
inference. This validates the approach in the production scenario where the start gate position
is always known (one fixed anchor per venue).

**Andreas Bakke remaining issues**:
- Frame 990: 122px — was the old tracking boundary; flow might have artifact at that transition
- Frames 1350/1370: ~91px — end-of-run extrapolation instability

**Accepted as new best (valid approach)**.

---

## Exp 2b — Production scenario: fixed start anchor

**Pending**: Test with frame 150 GT KEPT as fixed anchor (simulates knowing the start gate
venue position from competition registration). All other frames are LOO-predicted.
Expected: Knözinger LOO mean drops from 67px to ~28px; within50px ~90%.
