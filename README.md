# AI Judging — 3D Pose Estimation for Freeride Skiing

Markerless 3D pose estimation pipeline that converts monocular video of skiers
into 3D skeletal models.

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
│  2. Person Segmentation   │  (YOLOv11x-Seg + ByteTrack, imgsz=1280)
└─────────┬────────────────┘
          ▼
┌──────────────────────────┐
│  3. Temporal Tracking     │  (multi-track merge, OF-guided identity,
│                           │   Kalman CA+RTS smoothing)
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
```

## Setup

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install core dependencies
uv sync

# Full pipeline on a single video
uv run python -m src.pipeline raw_videos/SOME_VIDEO.mp4

# Process all videos (with a cap)
uv run python -m src.pipeline --max-videos 3

# Quick test mode (5 fps, 50 frames max)
uv run python -m src.pipeline --test raw_videos/SOME_VIDEO.mp4

# Edit config
$EDITOR config.yaml
```

### Running Individual Pipeline Stages

Use the `--stage` flag to run a single stage independently on an existing video. This is useful for comparing performance with different hyperparameter settings without re-running upstream stages.

**Available stages:**

- `frames` — Extract frames from video
- `segmentation` — Run YOLO person segmentation on frames
- `tracking` — Temporal tracking (requires segmentation output)
- `pose_2d` — 2D pose estimation (requires segmentation or tracking output)
- `pose_3d` — 3D pose lifting (requires 2D pose output) *[disabled]*
- `visualization` — Regenerate visualizations (requires frames + poses_2d)

**Examples:**

```bash
# Run only tracking stage (reuses existing segmentation output)
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage tracking

# Run only 2D pose estimation with different config
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage pose_2d

# Test frame extraction in quick mode
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage frames --test

# Regenerate visualizations with different settings
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage visualization
```

Each stage validates that its required upstream outputs exist and will fail with a clear error if dependencies are missing.

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
│   └── visualizations/       # Overlay videos & 3D HTML
└── src/
    ├── pipeline.py           # Main orchestrator
    ├── segmentation/         # YOLO person segmentation + ByteTrack IDs
    ├── tracking/             # Track selection, gap filling, bbox smoothing
    ├── pose_2d/              # 2D pose estimation (YOLO-Pose)
    ├── pose_3d/              # 2D → 3D lifting (MotionBERT stub)
    ├── visualization/        # 3D rendering & overlays
    └── utils/video.py        # Frame extraction
```

## Temporal Skier Tracking

The pipeline uses a multi-stage approach to keep a consistent identity for the
skier across all frames, even when the detector briefly loses track or ByteTrack
fragments the skier into multiple short-lived track IDs.

### Stage 2 — Segmentation with ByteTrack

`segment_skier()` in `src/segmentation/yolo_seg.py` calls Ultralytics
`model.track(persist=True, tracker="bytetrack.yaml")` instead of plain
`model()`.  ByteTrack assigns a persistent integer **track ID** to each
detected person across frames.  The segmentation manifest stores *all*
detected persons per frame (with their track IDs), not just the selected one,
so the tracking stage has full information to work with.

**Input resolution (`imgsz`)** is set to **1280** (up from the default 640).
This doubles the effective resolution, significantly improving detection of
small/distant skiers. In testing, this reduced zero-detection frames from 37%
to 30% on the hardest video and improved tracking accuracy across all videos.

### Stage 3 — Track Selection, Merge & Smoothing

`track_skier()` in `src/tracking/tracker.py` processes the raw detections
in six phases:

#### Phase A — Multi-Track Scoring & Merge

Each ByteTrack ID is scored with a weighted composite:

```
score = w_conf × mean_confidence
      + w_center × center_proximity
      + w_length × track_length_ratio
```

All tracks above `merge_score_threshold` are selected for merging to handle
track fragmentation (same skier split across multiple track IDs due to
occlusion or detection drops).

A **multi-candidate optical-flow trace** is built to guide merge conflict
resolution. Unlike single-track seeding, this OF trace can re-anchor to the
closest candidate detection from *any* track within a distance threshold
(`max_reanchor_dist_px=150px`), allowing it to seamlessly follow the skier
across ByteTrack track-ID boundaries.

When multiple candidates exist for a frame, the merge resolver scores each
candidate on six axes:

| Signal | Weight | Description |
|--------|--------|-------------|
| **Quality** | `w_conf + w_center` | Detection confidence + center proximity |
| **Continuity** | `w_continuity=0.6` | Proximity to predicted position (previous bbox + OF displacement) |
| **Track stickiness** | `w_track_stickiness=0.4` | Bonus for same ByteTrack ID as previous frame |
| **OF agreement** | `w_of_agreement=1.5` | Proximity to independent OF trace prediction |
| **Velocity consistency** | `w_velocity=0.4` | Penalises sudden velocity changes vs recent motion history |
| **Synthetic OF candidate** | `of_synthetic_confidence=0.3` | Injects a virtual candidate at the OF position when no YOLO detection is nearby |

During **genuine conflicts** (candidate spread > 100px with OF available),
the OF agreement weight is tripled (×3) and stickiness is reduced (×0.2),
preventing the tracker from locking onto the wrong person during crossover
events.

Candidates further than `2× of_tight_radius_px` (300px) from the OF trace
are pre-filtered before scoring. When **no** candidate passes this filter,
a **synthetic OF candidate** is injected at the OF-predicted position using
the previous bbox dimensions (`confidence=0.3`). This prevents the tracker
from defaulting to a bystander when the skier is mid-air and undetected by YOLO.
The OF agreement score is zeroed for synthetic candidates to avoid circular
self-reinforcement.

#### Phase A.6 — Identity Guard

An independent OF trace is maintained and detections that jump too far from
the predicted position are rejected (`max_jump_px=150`). The trace is
periodically re-anchored to confident detections to prevent drift.

#### Phase B — Gap Filling with Optical Flow

Every frame is guaranteed exactly one bounding box. Detection gaps are filled
using **optical flow** to propagate motion between frames:

- **Sparse Lucas-Kanade flow** (preferred): tracks feature points inside the
  bbox region, computes median displacement with outlier rejection via MAD.
- **Dense Farneback flow** (fallback): computes pixel-level motion vectors
  when sparse tracking has insufficient keypoints.
- **Bidirectional blending** for internal gaps: forward and backward flow
  predictions are linearly blended across the gap.
- **Anchor propagation** for leading/trailing gaps: flow is propagated up to
  `flow_max_extrapolate_frames` (30) frames; beyond that, the anchor bbox is
  copied.

#### Phase C — Kalman CA+RTS Smoothing

A **constant-acceleration Kalman filter** with **Rauch-Tung-Striebel (RTS)
backward smoother** replaces the earlier EMA smoothing. The state vector is
`[cx, cy, vx, vy, ax, ay, w, h]`. Detected frames receive lower measurement
noise than interpolated frames. Optical flow velocities are incorporated as
velocity measurements when available.

#### Phase D — Crop Generation & Manifest

Padded bounding boxes are used to extract crops for downstream 2D pose
estimation. Every frame has exactly one crop. The tracking manifest
(`tracking_manifest.json`) records per-frame fields:

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | int | Frame index |
| `frame_file` | str | Source frame filename |
| `detected` | bool | True if from a YOLO detection |
| `interpolated` | bool | True if from OF gap-filling |
| `of_synthetic` | bool | True if from a synthetic OF candidate |
| `bbox` | [x1,y1,x2,y2] | Tight bounding box |
| `bbox_padded` | [x1,y1,x2,y2] | Padded bbox used for crop |
| `confidence` | float | Detection confidence (0.3 for synthetics) |
| `track_id` | int | ByteTrack ID (−1 for interpolated) |
| `crop_file` | str | Path to the extracted crop image |
| `flow_displacement` | [dx,dy] | Optical flow displacement vector |
| `of_predicted` | [cx,cy] | OF-predicted center position |
| `velocity` | [vx,vy] | Frame-to-frame velocity vector |
| `speed_px_per_frame` | float | Scalar speed (pixels/frame) |

Top-level manifest fields include `velocity_stats` (aggregate `mean_speed`,
`max_speed`, `mean_vy`) and `identity_guard_rejections` (count of frames
rejected by the identity guard).

### Tracking Accuracy

Evaluated against hand-annotated ground truth on 3 competition videos
(center-point annotations every ~10 frames, linearly interpolated):

| Video | Mean Error (px) | HOTA |
|-------|:---:|:---:|
| Arno Vuarnier | 42.4 | 0.756 |
| Andreas Bakke | 18.2 | 0.896 |
| Lach Powell | 36.9 | 0.880 |
| **Overall** | **32.5** | **0.844** |

Key improvement history:
- Baseline (single-track, EMA): 107.4px mean, 0.710 HOTA
- OF-weighted merge resolution: 64.7px, 0.760 HOTA
- OF candidate pre-filtering: 62.5px, 0.766 HOTA
- imgsz=1280 + multi-candidate OF trace: 39.5px, 0.836 HOTA
- **Synthetic OF candidates: 32.5px, 0.844 HOTA**

### Gap Filling with Optical Flow

When the selected skier track has detection gaps (frames where the detector
lost the person), the tracking system must estimate a bounding box.  By
default, gaps are filled using **optical flow** to propagate motion between
frames—preserving natural skier movement trajectories rather than crude
linear interpolation.

**Fallback to linear interpolation**: If optical flow is disabled via
`optical_flow_method: "none"` or numpy is unavailable, gaps are filled
with simple linear interpolation of bbox coordinates.

### Tracking Evaluation & Ground Truth

Three videos have hand-annotated center-point ground truth in
`annotations/tracking/<video_stem>/gt_centers.csv`. Evaluate with:

```bash
# Evaluate all annotated videos (reports per-video + aggregate metrics)
uv run python -m src.tracking.evaluate --batch

# Render GT overlay videos (green=GT, red=predicted, with error HUD)
uv run python -m src.tracking.overlay_gt --batch
```

### Configuration

All tracking parameters live in the `tracking:` section of `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable the tracking stage |
| `w_conf` | `0.3` | Weight for mean detection confidence |
| `w_center` | `0.5` | Weight for center-of-frame proximity |
| `w_length` | `0.2` | Weight for track duration ratio |
| `min_track_frames` | `10` | Ignore tracks shorter than this |
| `smooth_window` | `5` | Kalman smoother window (0 = off) |
| `merge_tracks` | `true` | Merge multiple high-scoring tracks |
| `merge_score_threshold` | `0.3` | Minimum score for track inclusion in merge |
| `optical_flow_method` | `"auto"` | `"auto"` (sparse→dense), `"sparse"`, `"dense"`, or `"none"` |
| `flow_min_keypoints` | `5` | Min tracked features for sparse LK before fallback to dense |
| `flow_max_extrapolate_frames` | `30` | Max frames to propagate via OF for leading/trailing gaps |
| `identity_guard_enabled` | `true` | Hybrid OF identity guard |
| `identity_guard_max_jump_px` | `150` | Reject detection if too far from OF prediction |
| `identity_guard_reanchor_interval` | `50` | Re-anchor OF seed every N confident detected frames |
| `identity_guard_reanchor_min_conf` | `0.5` | Minimum detection confidence for re-anchor |
| `identity_guard_max_drift_px` | `200` | Force re-anchor if OF drift exceeds this |
| `w_velocity` | `0.4` | Weight for velocity-consistency scoring (0 = disabled) |
| `vel_history_len` | `5` | Number of recent velocity vectors to maintain |
| `of_synthetic_confidence` | `0.3` | Confidence for synthetic OF candidates (0 = disabled) |
| `cmc_enabled` | `false` | Enable camera motion compensation for gap filling |
| `cmc_method` | `"ecc"` | CMC method: `"orb"`, `"ecc"`, or `"none"` |
| `cmc_exclude_margin` | `1.5` | Exclude N× skier bbox area from feature matching |
| `cmc_min_features` | `20` | Minimum matched features for valid homography |
| `cmc_ransac_threshold` | `3.0` | RANSAC outlier threshold in pixels |

Set `tracking.enabled: false` to bypass tracking entirely and fall back to
the per-frame segmentation selection (previous behaviour).

## Model Notes

| Stage | Model | Config | Why |
|-------|-------|--------|-----|
| Person Segmentation | **YOLOv11x-seg** (Ultralytics) | `imgsz=1280` | Pixel-level instance masks for clean crops; higher resolution improves small-person recall |
| 2D Pose | **YOLO11x-Pose** (Ultralytics) | `imgsz=640` | Single-model 17-keypoint detection, MPS-native |
| 3D Lift | **MotionBERT** | — | Temporal transformer, state-of-the-art monocular 3D |

The pipeline includes fallback stubs so you can run the end-to-end flow even before downloading model checkpoints.
