# AI Judging — 3D Pose Estimation for Freeride Skiing

Markerless 3D pose estimation pipeline that converts monocular video of skiers
into 3D skeletal models (including ski detection).

## Pipeline

```
raw_video.mp4
    │
    ▼
┌──────────────────────┐
│  1. Frame Extraction  │  (OpenCV)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  2. Person Segment.   │  (YOLOv11-seg) ← isolate skier
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  3. 2D Pose (17 kp)  │  (RTMPose / MMPose)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  4. 3D Lifting       │  (MotionBERT)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  5. Ski Detection    │  (GroundingDINO + SAM2 / colour seg)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  6. Visualization    │  (Plotly 3D / OpenCV overlay)
└──────────────────────┘
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
│   ├── segmentation/         # YOLO crops & masks
│   ├── poses_2d/             # 2D keypoint JSONs
│   ├── poses_3d/             # 3D keypoint JSONs
│   ├── ski_masks/            # Ski segmentation masks
│   └── visualizations/       # Overlay videos & 3D HTML
└── src/
    ├── pipeline.py           # Main orchestrator
    ├── segmentation/         # YOLO person segmentation
    ├── pose_2d/rtmpose.py    # 2D pose estimation
    ├── pose_3d/lifter.py     # 2D → 3D lifting
    ├── ski_detection/        # Ski detection module
    ├── visualization/        # 3D rendering & overlays
    └── utils/video.py        # Frame extraction
```

## Model Notes

| Stage | Model | Why |
|-------|-------|-----|
| Segmentation | **YOLOv11x-seg** (Ultralytics) | Fast instance segmentation to isolate the skier from background/spectators |
| 2D Pose | **RTMPose-X** (MMPose) | Best accuracy/speed trade-off for real-time body keypoints |
| 3D Lift | **MotionBERT** | Temporal transformer, state-of-the-art monocular 3D |
| Ski Det | **GroundingDINO + SAM2** | Zero-shot prompted detection ("ski") with pixel-precise masks |

The pipeline includes fallback stubs so you can run the end-to-end flow
even before downloading model checkpoints.
