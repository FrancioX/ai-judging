# Experiment Loop: Venue Mapping — 2026-03-14

## Goal
Make the video-to-venue-image position system work. The current LoFTR-based feature matching approach produces 538px mean error on the single annotated video (Ski Men_2_89_Andreas Bakke) — essentially random placement. The system must reliably project the tracked athlete center from video pixel coordinates to venue image coordinates.

## Focus Area
`scripts/venue_mapping.py` — the video-to-venue projection pipeline.

## Baseline
- **Video**: Ski Men_2_89_Andreas Bakke (only annotated video)
- **GT**: 15 annotated keyframes (frames 150–1370) in `annotations/venue/Ski Men_2_89_Andreas Bakke/gt_venue.csv`
- **Current mean error**: 538.3 px (LoFTR homography — essentially random)
- **Tracking note**: tracking output covers frames 150–990 only; 10 of 15 GT frames have tracking centers available

## Key Insight
We have 10 direct correspondences between:
- video tracking center (cx, cy) at GT-annotated frames
- ground-truth venue position (vx, vy) at those same frames

This is much better than feature-matching the raw video frame to the venue image in snow. The athlete-tracking trajectory in video pixel space is a proxy for their path through the venue.

## Candidate List (ordered by expected ROI)

1. **Tracking-to-venue homography** — fit a perspective transform (or affine) from video-center (cx,cy) → venue (vx,vy) using the 10 overlapping GT correspondences. Project all other frames using their tracking center + this transform. No LoFTR at all.

2. **Piecewise-linear temporal interpolation in venue space** — given sparse GT keyframes, interpolate venue position as a function of frame index (with tracking-derived scale to refine between anchors).

3. **Local homography per segment** — fit separate transforms between successive GT keyframes (each using ~2 correspondences + stability priors) to handle camera/perspective changes mid-run.

4. **LoFTR fix: athlete-centered crops** — instead of matching full video frame to full venue image, crop around the athlete in the video frame and match to the nearby region in the venue image (guided by prior knowledge of venue extent).

5. **SIFT + RANSAC as LoFTR replacement** — classical feature matching, typically more robust on textured scenes with partial snow.

6. **Homography + tracking residual refinement** — after fitting global homography, use per-frame tracking motion to refine positions between GT anchors.

## Evaluation
Single video only (only one annotated):
```bash
uv run python scripts/evaluate_venue_mapping.py --batch
```
Metric: `mean_error_px` (primary), `pct_within_50px` (secondary).
