---
name: evaluate-tracking
description: 'Evaluate tracking outputs against ground-truth annotations and compare metrics between experiments. Use for single-video or --batch evaluation after each tracking run.'
argument-hint: 'Choose batch or single-video mode and list baseline comparison metrics.'
user-invocable: true
---

# Evaluate Tracking

## When To Use
- Measure quality after each tracking experiment.
- Compare error and HOTA against baseline.
- Validate that speed optimizations do not break tracking quality.

## Procedure
1. Run batch evaluation across all annotated videos:

```bash
uv run python -m src.tracking.evaluate --batch
```

2. Run single-video evaluation when debugging a regression:

```bash
uv run python -m src.tracking.evaluate \
  "output/tracking/<video_stem>" \
  "annotations/tracking/<video_stem>/gt_centers.csv"
```

3. Record and compare at least:
- `mean_error_px`
- `median_error_px`
- `p90_error_px`
- `p95_error_px`
- `pct_within_threshold`
- `detection_rate`
- `HOTA`

## Notes
- Batch mode uses all available `annotations/tracking/*/gt_centers.csv`.
- Lower pixel error is better; higher HOTA is better.
