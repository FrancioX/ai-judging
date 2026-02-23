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
│  4. 2D Pose (17 kp)      │  (YOLO-Pose / RTMPose)
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

# 2. Install the OpenMMLab stack (mmcv builds C++ extensions — takes ~5 min)
bash scripts/install_mmlab.sh

# Process a single video (use --no-sync to keep pip-installed mmlab packages)
uv run --no-sync python -m src.pipeline raw_videos/SOME_VIDEO.mp4

# Process all videos (with a cap)
uv run --no-sync python -m src.pipeline --max-videos 3

# Edit config
$EDITOR config.yaml
```

> **Note:** Always use `uv run --no-sync` instead of `uv run`. Plain `uv run`
> re-syncs the venv and will remove the mmlab packages (mmcv, mmdet, mmpose)
> that were installed via `pip --no-build-isolation` in step 2.

### Why the two-step install?

The OpenMMLab packages (mmcv, mmdet, mmpose) have broken build metadata —
they depend on `pkg_resources` and `numpy` at build time without declaring them.
uv cannot resolve them in isolation, so `scripts/install_mmlab.sh` builds them
with `pip --no-build-isolation` using the already-installed setuptools (<70),
cython, and numpy from the venv. Core deps are pinned with `setuptools>=69,<70`
in `pyproject.toml` to ensure `pkg_resources` stays available.

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
    ├── pose_2d/              # 2D pose estimation (YOLO-Pose / RTMPose)
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

Set `tracking.enabled: false` to bypass tracking entirely and fall back to
the per-frame segmentation selection (previous behaviour).

## Model Notes

| Stage | Model | Why |
|-------|-------|-----|
| Person Detection | **RTMDet-m** (MMDet, inside MMPose) | Fast COCO person detector; provides bounding boxes for RTMPose |
| 2D Pose | **RTMPose-X** (MMPose) | Best accuracy/speed trade-off for real-time body keypoints |
| 3D Lift | **MotionBERT** | Temporal transformer, state-of-the-art monocular 3D |
| Ski Det | **GroundingDINO + SAM2** | Zero-shot prompted detection ("ski") with pixel-precise masks |

The pipeline includes fallback stubs so you can run the end-to-end flow
even before downloading model checkpoints.

### Re-enabling YOLO pixel segmentation

The previous pipeline used **YOLOv11x-seg** (Ultralytics) as a separate person
segmentation step _before_ pose estimation.  YOLO provides pixel-level instance
masks (not just bounding boxes), which can be useful for:

- **Background removal** — feeding cleaner crops to the pose estimator.
- **Downstream mask-based ski detection** — isolating the skier before
  colour/shape analysis.

Since RTMDet (built into MMPose) already handles person detection with
comparable accuracy, the YOLO step was removed from the default pipeline to
avoid running two detectors.  The YOLO code is fully preserved in
`src/segmentation/yolo_seg.py` and can be re-introduced by:

1. Calling `segment_skier()` between frame extraction and pose estimation.
2. Passing the YOLO crops + `segmentation_manifest` to `estimate_2d_poses()`.
3. The `--segment-only` CLI flag still works for standalone YOLO testing.

See the `segmentation` section in `config.yaml` for the YOLO settings.
