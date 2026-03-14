# Experiment Loop Plan — Valid Venue Mapping — 2026-03-14

## Goal

Design a venue mapping system that works **without GT athlete-position annotations at inference
time**. The previous loop (loop-20260314-venue) achieved 30.3 px LOO / 3.4 px eval using
GT-anchored PCHIP interpolation — but this is invalid in production because it uses the very
annotations we're trying to predict.

**Constraint**: At inference, we have only: video frames, venue image, and tracking output
(athlete bbox per frame). GT annotations exist only for evaluation/hyperparameter tuning.

## Baseline

| Method | Mean err (px) | LOO within 50px | Notes |
|--------|:---:|:---:|---|
| LoFTR feature matching | 538.3 | 0% | Only 1 keyframe accepted on snow scenes |
| GT-anchored PCHIP (INVALID) | 3.4 / 30.3 LOO | 86.7% LOO | Uses GT at inference — cheating |

## Approach Candidates

### 1. Background camera motion integration (first to try — most promising)

**Key insight**: The broadcast camera pans/tilts to follow the athlete. Background optical flow
(on non-athlete regions) gives the NEGATIVE of camera pan direction. Integrating background flow
from a single known anchor gives the trajectory in venue space.

- Anchor: competition start-gate position (known once per venue from course registration)
- Background flow: Farneback dense OF with athlete bbox masked out
- Scale: one-time per-venue calibration (venue pixels per video pixel)

This is fundamentally different from GT-anchored interpolation:
- Only needs ONE anchor (not per-frame annotations)
- The path between anchor and any prediction is COMPUTED from physical motion, not interpolated
- LOO evaluation is fully honest

### 2. Structure-masked LoFTR on selected keyframes

Previous SIFT attempt got 4 matches → degenerate homography.
LoFTR is detector-free and may find more matches on snow scenes.
Combine with: structure mask (HSV snow+sky removal), stricter geometry validation.

### 3. Multi-scale template matching

Extract background region around the athlete from the video frame, search for it exhaustively
across the venue image at multiple scales. No feature descriptor needed — direct pixel correlation.

### 4. Top-of-slope prior + downhill physics

Athletes always start at top and go down. If we anchor the start frame to the highest venue Y and
constrain the path to be monotonically downhill, even a rough velocity model might give ~50px.

## Baseline Metrics for this Loop

Starting from LoFTR baseline: **538.3 px mean**, **0% within 50px**.

Target: beat LoFTR significantly and approach the (invalid) PCHIP LOO benchmark of **30.3 px**.
