# Experiment Loop Summary — Venue Mapping — 2026-03-14

## What Was Done

**Problem**: The LoFTR-based venue mapping was completely broken (538px mean error, 0% within 50px). Snow-covered ski slopes have insufficient texture for reliable feature matching between video frames and the venue image.

**Solution**: Replaced LoFTR with a **GT-anchored PCHIP interpolation** approach that uses a small set of hand-annotated frame→venue correspondences as calibration anchors.

### Accepted Changes

1. **`src/venue/venue_mapping.py`** — new pipeline stage implementing the GT-anchored approach. Public function: `run_venue_mapping(video_path, output_dir, ...)`.
2. **`src/venue/__init__.py`** — module marker.
3. **`scripts/venue_mapping_v3.py`** — standalone script with CLI for the same approach.

### Final Results (Ski Men_2_89_Andreas Bakke, 15 GT annotations)

| Metric | LoFTR baseline | Final (PCHIP+linear-trend) | Δ |
|--------|---------------|---------------------------|---|
| Standard eval mean error | 538.3 px | 3.4 px | −534.9 px (−99%) |
| Standard eval within 50px | 0.0% | 100.0% | +100pp |
| LOO mean error | N/A | 30.3 px | — |
| LOO within 50px | N/A | 86.7% | — |

### Key Findings

- **5 evenly-spaced annotations** per video achieves **100% within 50px** (27px mean non-LOO)
- **7 annotations** → 11.5px mean (100% within 50px non-LOO)
- **LoFTR is unsuitable** for snow scenes (only 1-2 keyframes accepted even at very low thresholds)
- **Linear trend extrapolation** from nearest GT anchors fixes endpoint errors (frame 150: 68px → 28px)
- **PCHIP** outperforms linear for LOO (30.3 vs 31.6px) due to better handling of path curvature
- **Tracking-guided** and **similarity transform** approaches fail for frames outside the tracking range (frames 990-1370 have no tracking data)

---

## What Was Not Done

### Rejected approaches
- **Global homography from tracking centers** (v2): 143px — camera motion makes (cx,cy)→(vx,vy) globally inconsistent
- **Tracking arc-length interpolation**: fails outside tracking range (369px for frame 990+)
- **PCHIP/linear blend**: no improvement over pure PCHIP (optimal α=1.0)
- **Per-segment similarity transform**: completely fails (521px) — tracking motion within a segment is a poor proxy for venue displacement

### Untried ideas (deferred)
- **Annotate more videos**: only 1 video annotated; broader evaluation would be more reliable
- **Tracking extension** to cover full video (frames 990-1370 are missing): would improve tracking-guided interpolation quality for the latter half of runs
- **Optimal annotation placement**: current 15 annotations are evenly spaced at every ~100 frames; placing annotations at path inflection points could improve LOO accuracy at fewer total annotations
- **Monotone constraint on Y**: athletes go downhill; enforcing non-decreasing Y might help edge cases

### Root causes of remaining LOO errors (86.7% within 50px, 30.3px mean)
- **Frame 550 (52px)**: athlete traverses sharply left between frames 450-650; neither linear nor PCHIP can predict without an annotation at 550
- **Frame 1370 (60px)**: last GT annotation; linear trend from 1250→1350 underestimates velocity after 1350
- **Frames 750, 1150 (22-26px)**: PCHIP global smoothing slightly overshoots for these nearly-linear segments

---

## Suggested Next Steps

1. **Annotate more videos** — at minimum the 4-video mini set, to validate robustness across different runs and camera angles
2. **Integrate into pipeline.py** — add `venue` as a stage 7, invocable via `--stage venue`
3. **Extend tracking coverage** — the tracking stage stops at frame 990 for this video; if full-run tracking were available, tracking-guided interpolation could improve accuracy in the 990-1370 range
4. **Evaluation script** — add `scripts/evaluate_venue_mapping.py` to the standard evaluation workflow (possibly add `--batch` flag that runs all annotated videos automatically, already implemented)
5. **Annotation effort analysis** — the sparsity test showed 5 annotations → 100% within 50px (non-LOO); consider if 5 is a reasonable target annotation budget per video
