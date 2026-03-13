---
name: visualize-tracking
description: 'Render tracking versus ground-truth overlay videos for visual debugging of identity switches and drift. Use after evaluation to inspect failure modes frame by frame.'
argument-hint: 'Choose batch or single-video overlay generation.'
user-invocable: true
---

# Visualize Tracking

## When To Use
- Inspect why metrics changed after an experiment.
- Validate identity consistency visually.
- Spot drift, jumps, or delayed recovery not obvious in aggregate metrics.

## Procedure
1. Render overlays for all annotated videos:

```bash
uv run python -m src.tracking.overlay_gt --batch
```

2. Render one video overlay:

```bash
uv run python -m src.tracking.overlay_gt \
  "output/tracking/<video_stem>" \
  "annotations/tracking/<video_stem>/gt_centers.csv"
```

3. Inspect `output/tracking/<video_stem>/overlay_gt.mp4`:
- Green: ground truth
- Red: prediction
- Yellow vector: per-frame error
- HUD: running mean error

## Notes
- Use overlays as a qualitative check to complement quantitative metrics.
- Prioritize videos with largest HOTA or error deltas.
