# Experiment Loop Summary — Valid Venue Mapping — 2026-03-14/15

## Goal recap

Build a venue mapping system that works **without GT athlete-position annotations at inference
time**. The previous loop (loop-20260314-venue) achieved 30.3px LOO using GT-anchored PCHIP —
but that was invalid in production because it looked up GT positions directly as predictions.

---

## What was done (accepted changes)

### Core algorithm: background OF + PCHIP (venue_camflow.py)

**Algorithm**: Farneback dense optical flow on dark+textured background pixels (rocks/trees,
V<80 AND gradient>5); cumulative sum gives camera pan trajectory; PCHIP spline maps
cumulative flow → venue position. GT is used **only** for scale calibration, never as a
prediction lookup.

**Production scenario**: One fixed anchor per venue (start gate position, registered once
at competition setup) is all that is required. All other positions are predicted from
background flow integration.

**Final results** (Knözinger, 2343 frames, 25 GT annotations):

| Scenario | n_eval | LOO mean | LOO median | LOO P90 | LOO within 50px |
|----------|:---:|:---:|:---:|:---:|:---:|
| Unconstrained LOO | 25 | 67.0px* | 25.6px | 62.4px | 88% |
| Fixed anchor (frame 150) | 24 | **28.5px** | **25.2px** | **45.1px** | **92%** |

*67px mean is inflated by a single extrapolation failure (frame 150 = first GT point; when
held out, PCHIP must extrapolate to cum_flow=0 outside its calibration range → 989px).
In production this point is always known, so the fixed-anchor scenario is the correct metric.

**Andreas Bakke** (1223 frames, 15 GT): 52.9px LOO mean, 48.2px median, 53% within 50px.
The weaker result reflects fewer GT annotations, more zoom variation at frame 990, and
end-of-run extrapolation at frames 1350/1370. The GT annotation noise floor is ~20-30px
(annotations are visual clicks on a wide-angle image), so errors below ~25px are not
meaningful to optimize.

**Key engineering decisions**:
- Dark-feature OF (V<80, gradient>5) used in 100% of frame transitions for both videos —
  enough rock/tree features are visible in every frame.
- `zoom_correct=False`: the pipeline uses `fixed_crop_width=192, fixed_crop_height=320`
  so bbox area is approximately constant regardless of camera zoom; no zoom correction needed.
- PCHIP parameterised by `cum_flow_y` (monotone) rather than frame index — handles
  non-linear zoom variation that linear regression cannot.

### Video output (venue_camflow.py)

Added `_write_venue_video`: side-by-side MP4 (video frame left, venue image right with
blue athlete dot, fading trail, yellow GT crosses). Uses full-GT calibration for the video.

Output videos generated:
- `output/venue_mapping/Snowboard Men_1_80_Fabian Knözinger/camflow_venue.mp4`
- `output/venue_mapping/Ski Men_2_89_Andreas Bakke/camflow_venue.mp4`

---

## What was not done / rejected

### Exp 1 — Linear regression on background OF
Rejected. Scale varies non-linearly with camera zoom; linear regression gives 149-160px LOO.
PCHIP is necessary.

### Exp 3 — Multi-scale template matching + SIFT (automatic anchor)

Both approaches tried, both failed:

| Method | Result |
|--------|--------|
| NCC on gradient images (scale 0.03-0.55) | 735-771px mean error; small templates win, matching noise |
| SIFT on dark-feature regions | 34 keypoints/frame, 2 good matches — RANSAC fails |
| SIFT on full background | 182 keypoints/frame, 2 good matches — RANSAC fails |

**Root cause**: The broadcast camera and venue image are from **different physical viewpoints**,
not just different zoom levels. The same rocks and trees appear at different geometric angles in
the two images. Standard feature matching (NCC, SIFT, LoFTR) cannot bridge this gap:

- Snow dominates both images (94.7% of video frame, 80.7% of venue have near-zero gradient).
- The 5.3% dark+textured pixels in the video frame do not geometrically resemble the
  corresponding dark pixels in the venue image due to viewpoint difference.
- NCC is not discriminative enough: uniform snow makes every region look similar at the
  gradient level; small templates always win at minimum scale.

This is a **fundamental limitation** of the cross-view matching problem for ski slope footage.

---

## What would be needed for a fully automatic anchor

The one fixed anchor per venue (start gate position) is the most practical production
approach. It requires one human click per venue per competition day. Below are the realistic
paths toward eliminating even that:

1. **Venue image from the same camera**: If a wide-angle reference image is captured with
   the broadcast camera (just zoomed fully out) at competition start, it shares the same
   viewpoint as the video. Template matching would work precisely in this case. One image
   per venue per day.

2. **Slope silhouette / sky-horizon matching**: The boundary between the sky and the
   snow slope is a distinctive 1D profile visible from both viewpoints. When sky is present
   in the frame (first half of most runs), matching the silhouette shape against the venue
   image at multiple scales may localise the frame to ±50px. Not tried; requires sky to be
   consistently visible.

3. **GPS / terrain data**: Competition athletes carry timing chips; some venues have GPS
   data for athletes. If GPS coordinates can be registered to the venue image (a one-time
   geo-referencing step), every athlete position is automatically known.

4. **Finish line / course feature detection**: Detect a known competition structure
   (timing banner, safety net endpoint, a prominently coloured gate) in the video frame
   and in the venue image. Requires a specific detector per landmark but is robust once built.

---

## Suggested next steps

1. **Annotate more videos** — currently only 2 videos have venue GT. Annotate the full
   dev-set (10 videos) to get a reliable mean-LOO estimate and identify any videos where
   background OF fails (e.g., very dark frames, heavy snow).

2. **Try slope silhouette matching** — implement sky segmentation (bright+blue pixels) and
   extract the sky-slope boundary profile. Search this profile against the venue image edge
   at multiple horizontal offsets and scales. Most likely to work on early-run frames.

3. **Capture venue reference image from broadcast camera** — if possible at the next
   competition, take a single fully-zoomed-out image with the broadcast camera at the
   start of the day. This unlocks fully automatic template-based anchoring.

4. **Wire venue_camflow into the main pipeline** as stage 7 (after tracking), writing
   `output/venue_mapping/<stem>/camflow_venue.mp4` and a JSON manifest.
