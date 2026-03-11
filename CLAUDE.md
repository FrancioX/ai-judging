# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Markerless 3D pose estimation pipeline for freeride skiing competition judging. Converts monocular video → 3D skeletal model with ski detection.

## Commands

```bash
# Install dependencies
uv sync

# Run full pipeline
uv run python -m src.pipeline raw_videos/VIDEO.mp4
uv run python -m src.pipeline --test VIDEO.mp4       # quick: 5fps/50 frames
uv run python -m src.pipeline --max-videos 3         # batch

# Run a single stage (reuses upstream outputs)
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage tracking
# Available stages: frames, segmentation, tracking, pose_2d, pose_3d, ski_detection, visualization

# Tests & lint
uv run pytest
uv run ruff check src/

# Tracking evaluation
uv run python -m src.tracking.evaluate --batch
uv run python -m src.tracking.overlay_gt --batch

# Annotate a new video for ground truth
uv run python -m src.tracking.annotate_centers output/frames/<stem> annotations/tracking/<stem>/gt_centers.csv --step=10

# Interactive review of tracker predictions (frame-by-frame inspection)
uv run python -m src.tracking.review_tracks output/frames/<stem> output/tracking/<stem>/tracking.json
```

## Architecture

Seven-stage disk-coupled pipeline orchestrated by [src/pipeline.py](src/pipeline.py):

```
Raw Video → Frame Extraction → Person Segmentation (YOLOv11x-Seg + ByteTrack)
         → Temporal Tracking → 2D Pose (YOLO11x-Pose) → 3D Lifting (MotionBERT, stub)
         → Ski Detection (GroundingDINO+SAM2 / colour fallback, stub) → Visualization
```

Stages communicate via **JSON manifests** and image directories under `output/<stage>/<video_stem>/`. Each stage exposes one public function: `fn(input_path, output_dir, *, config_kwargs...) → Path`. See [src/segmentation/yolo_seg.py](src/segmentation/yolo_seg.py) as the canonical example.

Config is a plain `dict` loaded from [config.yaml](config.yaml) — accessed via `config.get("key", default)`. No Pydantic/dataclass schema.

**Import fallback pattern** (mandatory for ML stages): wrap heavy imports in `try/except ImportError` and fall back to a `_write_stub_*()` function so the pipeline runs end-to-end without all models installed.

### Tracking System

[src/tracking/tracker.py](src/tracking/tracker.py) is the core of the pipeline, operating in three phases:

- **Phase A — Multi-track merge**: scores each ByteTrack ID on confidence, center proximity, and track length; merges fragments. A 6-axis candidate scorer (confidence, continuity, track stickiness, OF agreement, velocity consistency, synthetic OF) resolves conflicts.
- **Phase A.6 — Identity guard**: an independent optical-flow trace rejects detections jumping >150px from its prediction (`identity_guard_max_jump_px`) to prevent ID switches.
- **Phase B — Gap filling**: Lucas-Kanade (preferred) or Farneback dense OF, with bidirectional blending for internal gaps.
- **Phase C — Kalman smoothing**: constant-acceleration Kalman filter + RTS backward smoother on state `[cx, cy, vx, vy, ax, ay, w, h]`.

All 24+ tunable tracking parameters live under the `tracking:` key in [config.yaml](config.yaml).

## Conventions

- **Python 3.12** with `from __future__ import annotations` in every module.
- PEP 604 unions (`str | None`, not `Optional[str]`). Absolute imports only (`from src.segmentation.yolo_seg import …`).
- Naming: `snake_case` functions (verb-first), `_prefixed` private helpers, `UPPER_SNAKE_CASE` constants.
- No custom exception classes — use `FileNotFoundError`, `RuntimeError`, `ImportError` with descriptive messages.
- Default device: `mps` (Apple Silicon). Support `cuda:0` and `cpu` via config.
- One public function per stage module. New stages go in `src/<stage_name>/<module>.py` with an empty `__init__.py`.

## Test Video Reference

When the user refers to "the test video" or "example video", they mean:
`VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89.mp4`

Pre-processed outputs for all stages are available in `output/`.

## Tracking Ground Truth

Videos have hand-annotated center-point annotations in `annotations/tracking/<video_stem>/gt_centers.csv` (format: `frame_id,track_id,center_x,center_y`; sparse ~every 10th frame, linearly interpolated during evaluation).

**After any tracking change**, evaluate against ground truth to catch regressions:

```bash
# 1. Baseline
uv run python -m src.tracking.evaluate --batch

# 2. Re-run tracking
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage tracking

# 3. Compare
uv run python -m src.tracking.evaluate --batch

# 4. Visual inspection
uv run python -m src.tracking.overlay_gt --batch
# Output: output/tracking/<stem>/overlay_gt.mp4 — green=GT, red=predicted
```

Current aggregate: **32.5px mean error, 0.844 HOTA**.

## Experiment Logging

When running tracking experiments, always log results in [experiments/experiment_log.md](experiments/experiment_log.md). Each entry must include: date, goal (one-sentence hypothesis), implementation summary, results table (per-video `mean_error_px` + `HOTA` with delta vs previous best), and conclusion. Use sub-entries (3a, 3b, …) for iterative tuning. Update the "Current Best" section when a new best is achieved.
