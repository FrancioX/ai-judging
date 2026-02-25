# AI Judging — 3D Pose Estimation for Freeride Skiing

Markerless 3D pose estimation pipeline that converts monocular video of skiers
into 3D skeletal models (including ski detection).

## Pipeline

```
raw_video.mp4
    │
    ▼
┌──────────────────────────┐
│  1. Frame Extraction      │  (OpenCV)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  2. Person Segmentation   │  (YOLO-Seg + ByteTrack)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  3. Temporal Tracking     │  (track selection, gap fill, EMA smooth)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  4. 2D Pose (17 kp)      │  (YOLO-Pose)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  5. Visualization         │  (OpenCV overlay / Plotly 3D)
└──────────────────────────┘

Disabled stages (re-enable when models are ready):
  • 3D Lifting (MotionBERT)
  • Ski Detection (GroundingDINO + SAM2)
```

## Setup

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install core dependencies
uv sync

# Process a single video
uv run python -m src.pipeline raw_videos/SOME_VIDEO.mp4

# Process all videos (with a cap)
uv run python -m src.pipeline --max-videos 3

# Edit config
$EDITOR config.yaml
```

## Configuration

All settings live in `config.yaml` — frame extraction rate, model selection,
device (`mps` for Apple Silicon, `cuda:0` for NVIDIA), thresholds, etc.

## Project Structure

```
ai_judging/
├── config.yaml              # Pipeline configuration
├── pyproject.toml            # uv project definition
├── raw_videos/               # Input .mp4 files
├── output/                   # Generated artifacts
│   ├── frames/               # Extracted video frames
│   ├── segmentation/         # YOLO crops, masks & per-frame detections
│   ├── tracking/             # Tracked crops, smoothed bboxes
│   ├── poses_2d/             # 2D keypoint JSONs
│   ├── poses_3d/             # 3D keypoint JSONs
│   ├── ski_masks/            # Ski segmentation masks
│   └── visualizations/       # Overlay videos & 3D HTML
└── src/
    ├── pipeline.py           # Main orchestrator
    ├── segmentation/         # YOLO person segmentation + ByteTrack IDs
    ├── tracking/             # Track selection, gap filling, bbox smoothing
    ├── pose_2d/              # 2D pose estimation (YOLO-Pose)
    ├── pose_3d/              # 2D → 3D lifting (MotionBERT stub)
    ├── ski_detection/        # Ski detection module
    ├── visualization/        # 3D rendering & overlays
    └── utils/video.py        # Frame extraction
```

## Temporal Skier Tracking

The pipeline uses a two-stage approach to keep a consistent identity for the
skier across all frames, even when the detector briefly loses track:

### Stage 2 — Segmentation with ByteTrack

`segment_skier()` in `src/segmentation/yolo_seg.py` calls Ultralytics
`model.track(persist=True, tracker="bytetrack.yaml")` instead of plain
`model()`.  ByteTrack assigns a persistent integer **track ID** to each
detected person across frames.  The segmentation manifest stores *all*
detected persons per frame (with their track IDs), not just the selected one,
so the tracking stage has full information to work with.

### Stage 3 — Track Selection, Gap Filling & Smoothing

`track_skier()` in `src/tracking/tracker.py` processes the raw detections
in three phases:

1. **Track selection** — Each ByteTrack ID is scored with a weighted
   composite:

   ```
   score = w_conf × mean_confidence
         + w_center × center_proximity
         + w_length × track_length_ratio
   ```

   *Center proximity* is `1 − d / d_max` where `d` is the Euclidean
   distance from the bbox centre to the frame centre.  This encodes the
   assumption that the target skier will generally be near the middle of the
   frame.  Tracks shorter than `min_track_frames` are discarded.  The
   highest-scoring track is selected as the skier.

2. **Gap filling** — Every frame is guaranteed to have exactly one
   bounding box.  For frames where the selected track has no detection,
   bbox coordinates are **linearly interpolated** between the nearest
   detected neighbours.  Leading/trailing gaps are filled by propagating
   the nearest anchor.  No frame is ever left without a bbox.

3. **Temporal smoothing** — A zero-phase **exponential moving average**
   (forward + backward EMA pass) is applied to the bbox coordinates to
   reduce frame-to-frame jitter.  The window size is configurable via
   `smooth_window` (set to `0` to disable).

The tracking stage writes its own crops and manifest
(`output/tracking/<video>/tracking.json`), which the downstream 2D pose
stage reads automatically.

### Gap Filling with Optical Flow

When the selected skier track has detection gaps (frames where the detector
lost the person), the tracking system must estimate a bounding box.  By
default, gaps are filled using **optical flow** to propagate motion between
frames—preserving natural skier movement trajectories rather than crude
linear interpolation.

**How it works:**

- **Sparse Lucas-Kanade flow** (preferred) detects and tracks feature
  points inside and around the bounding box region across two consecutive
  frames, computing a median displacement of reliable (inlier) points.
  Outliers are rejected via median absolute deviation.  If fewer than
  `flow_min_keypoints` features are successfully tracked, sparse flow
  falls back to dense flow.

- **Dense Farneback flow** (fallback) computes motion vectors for every
  pixel in a region around the bbox.  The mean flow within the original
  bbox region is used as the displacement estimate.

- **Bidirectional propagation** (internal gaps): Forward flow is computed
  from the left anchor and backward flow from the right anchor, then the
  two estimates are blended linearly across the gap.  This produces
  physically plausible intermediate positions.

- **Anchor propagation** (leading/trailing gaps): Leading frames (before
  the first detection) and trailing frames (after the last detection) are
  filled by repeatedly applying flow backward/forward up to
  `flow_max_extrapolate_frames` frames.  Beyond that distance, the nearest
  anchor bbox is copied to avoid unrealistic extrapolation.

- **Fallback to linear interpolation**: If optical flow is disabled via
  `optical_flow_method: "none"` or numpy is unavailable, gaps are filled
  with simple linear interpolation of bbox coordinates.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optical_flow_method` | `"auto"` | `"auto"` (sparse→dense), `"sparse"` (Lucas-Kanade only), `"dense"` (Farneback only), or `"none"` (linear interpolation) |
| `flow_min_keypoints` | `5` | Minimum tracked features before sparse LK falls back to dense |
| `flow_max_extrapolate_frames` | `30` | Maximum frames to propagate via flow for leading/trailing gaps |

**Velocity-aware smoothing:**

When optical flow is active, the per-frame flow displacement is recorded in
the manifest (`flow_displacement` field).  The temporal smoothing pass is
then **velocity-aware**: frames with high optical flow (fast-moving skier)
are smoothed less, while static frames are smoothed more.  This preserves
sharp directional changes during tricks while reducing jitter during slow
passages.

### Configuration

All tracking parameters live in the `tracking:` section of `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable the tracking stage |
| `w_conf` | `0.3` | Weight for mean detection confidence |
| `w_center` | `0.5` | Weight for center-of-frame proximity |
| `w_length` | `0.2` | Weight for track duration ratio |
| `min_track_frames` | `10` | Ignore tracks shorter than this |
| `smooth_window` | `5` | EMA window for bbox smoothing (0 = off) |
| `optical_flow_method` | `"auto"` | Optical flow method: `"auto"`, `"sparse"`, `"dense"`, or `"none"` |
| `flow_min_keypoints` | `5` | Min tracked features for sparse Lucas-Kanade before fallback to dense |
| `flow_max_extrapolate_frames` | `30` | Max frames to propagate via optical flow for leading/trailing gaps |

Set `tracking.enabled: false` to bypass tracking entirely and fall back to
the per-frame segmentation selection (previous behaviour).

## Model Notes

| Stage | Model | Why |
|-------|-------|-----|
| Person Segmentation | **YOLOv11x-seg** (Ultralytics) | Pixel-level instance masks for clean crops |
| 2D Pose | **YOLO11x-Pose** (Ultralytics) | Single-model 17-keypoint detection, MPS-native |
| 3D Lift | **MotionBERT** | Temporal transformer, state-of-the-art monocular 3D |
| Ski Det | **GroundingDINO + SAM2** | Zero-shot prompted detection ("ski") with pixel-precise masks |

The pipeline includes fallback stubs so you can run the end-to-end flow even before downloading model checkpoints.
