# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Markerless 3D pose estimation pipeline for freeride skiing competition judging. Converts monocular video → 3D skeletal model.

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
# Available stages: frames, segmentation, tracking, pose_2d, pose_3d, visualization

# Tests & lint
uv run pytest
uv run ruff check src/

# Tracking evaluation
uv run python -m src.tracking.evaluate --batch
uv run python -m src.tracking.overlay_gt --batch

# Annotate ground truth — from scratch
uv run python -m src.tracking.annotate_centers output/frames/<stem> annotations/tracking/<stem>/gt_centers.csv --step=10

# Annotate ground truth — review/correct existing tracker output
uv run python -m src.tracking.annotate_centers output/frames/<stem> annotations/tracking/<stem>/gt_centers.csv --from-track=output/tracking/<stem>
```

## Architecture

Six-stage disk-coupled pipeline orchestrated by [src/pipeline.py](src/pipeline.py):

```
Raw Video → Frame Extraction → Person Segmentation (YOLOv11x-Seg + ByteTrack)
         → Temporal Tracking → 2D Pose (YOLO11x-Pose) → 3D Lifting (MotionBERT, stub)
         → Visualization
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

## Development Video Subsets

Two smaller subsets for fast iteration during tracking experiments (instead of all 18 videos).

### 10-video dev set (5 ski + 5 snowboard)

Chosen to cover the full difficulty range (Exp 21a stats).

| Sport | Athlete | Video stem | Mean Err (px) | HOTA |
|-------|---------|------------|:---:|:---:|
| Ski | Andreas Bakke | `VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89` | 8.7 | 0.932 |
| Ski | Lach Powell | `VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86` | 9.2 | 0.931 |
| Ski | Emile Peizerat | `VERBIER FREERIDE WEEK QUALIFIER 4__11_Ski Men_Emile Peizerat_76_France_74.83_` | 15.7 | 0.858 |
| Ski | Jordan Koch | `VERBIER FREERIDE WEEK QUALIFIER 4__10_Ski Men_Jordan Koch_56_Switzerland_75_` | 42.6 | 0.777 |
| Ski | Gabin Leonard | `VERBIER FREERIDE WEEK QUALIFIER 4__12_Ski Men_Gabin Leonard_26_France_74_` | 58.5 | 0.867 |
| Snowboard | Jonatan Laland | `VERBIER FREERIDE WEEK QUALIFIER 4__16_Snowboard Men_Jonatan Laland_61_Norway_40_` | 4.6 | 0.955 |
| Snowboard | Adriano Cardillo | `VERBIER FREERIDE WEEK QUALIFIER 4__12_Snowboard Men_Adriano Cardillo_63_Switzerland_52.33_` | 9.8 | 0.913 |
| Snowboard | Cedric Giraudeau | `VERBIER FREERIDE WEEK QUALIFIER 4__10_Snowboard Men_Cedric Giraudeau_67_France_54.33_` | 13.1 | 0.906 |
| Snowboard | Quentin Puydenus | `VERBIER FREERIDE WEEK QUALIFIER 4__17_Snowboard Men_Quentin Puydenus_53_France_35_` | 47.8 | 0.753 |
| Snowboard | Theodor Salen | `VERBIER FREERIDE WEEK QUALIFIER 4__15_Snowboard Men_Theodor Salen_59_Norway_45_` | 54.6 | 0.777 |

### 4-video mini set (2 ski + 2 snowboard)

Maximum contrast: one easy + one hard per sport. Use for the fastest sanity-checks.

| Sport | Athlete | Video stem | Mean Err (px) | HOTA |
|-------|---------|------------|:---:|:---:|
| Ski | Andreas Bakke | `VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89` | 8.7 | 0.932 |
| Ski | Arno Vuarnier | `VERBIER FREERIDE WEEK QUALIFIER 4__1_Ski Men_Arno Vuarnier_58_Switzerland_91.33` | 47.9 | 0.734 |
| Snowboard | Jonatan Laland | `VERBIER FREERIDE WEEK QUALIFIER 4__16_Snowboard Men_Jonatan Laland_61_Norway_40_` | 4.6 | 0.955 |
| Snowboard | Quentin Puydenus | `VERBIER FREERIDE WEEK QUALIFIER 4__17_Snowboard Men_Quentin Puydenus_53_France_35_` | 47.8 | 0.753 |

## Experiment Logging

For **manual experiments** (outside of a loop), log results by appending an entry to [experiments/experiment_log.md](experiments/experiment_log.md). Each entry must include: date, goal (one-sentence hypothesis), implementation summary, results table (per-video `mean_error_px` + `HOTA` with delta vs previous best), and conclusion. Use sub-entries (3a, 3b, …) for iterative tuning. Update the "Current Best" section when a new best is achieved.

For **experiment loops**, each loop writes to its own dedicated directory — see the Experiment Loop section below.

## Experiment Loop

### Trigger
Start when the user says "run an experiment loop" or similar phrasing. The loop runs **indefinitely** until the user explicitly asks to stop.

### Setup
1. Create a new git worktree + branch named `exp-loop-YYYYMMDD` (today's date) and work exclusively inside it for the duration of the loop. Leave it in place when the loop ends so the user can review the full commit history.
2. Create a dedicated directory for this loop at `experiments/loop-YYYYMMDD/`. It must contain exactly three files, created at the times indicated:
   - `plan.md` — written at **loop start**: one-paragraph goal statement, focus area, candidate list, and baseline metrics.
   - `log.md` — written **incrementally**: all experiment entries appended after each iteration. Do not append to `experiment_log.md`.
   - `summary.md` — written at **loop end**: what was done (accepted changes + final metrics), what was not done (rejected/untried candidates with root-cause notes), and suggested next steps.
3. Note any context the user provides at startup:
   - **Focus area** — which pipeline stage to improve (e.g., tracking).
   - **Candidate list** — potential improvements to try.
   - **Experiment history** — experiments already run (from `experiments/experiment_log.md`, existing loop logs, or supplied inline).

### Per-Iteration Steps

**1. Select**
Pick the most promising untried improvement from the candidate list. Do not repeat a previously discarded experiment unless a new change creates a clear synergy that did not exist when it was first tried.

**2. Implement**
Make the change. Prefer improvements that are a single self-contained step, or that can be decomposed into independent steps. Avoid changes that greatly increase coupling between components or algorithmic complexity. A modular improvement that is slightly weaker beats a tightly coupled one that is slightly stronger.

**3. Quick evaluation — mini set**
Run the stage being tested on the 4-video mini set and evaluate:
```bash
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage tracking  # repeat for all 4 mini-set videos
uv run python -m src.tracking.evaluate --batch
```

**4. Full evaluation — dev set** *(only if mini set shows improvement)*
Run on the full 10-video dev set and evaluate to get a reliable score estimate.

**5. Decide**

| Outcome | Action |
|---------|--------|
| Metrics improve | Accept — keep code change, update `config.yaml` to the winning configuration |
| Metrics regress or neutral | Reject — revert code change, restore `config.yaml` to current best |

**6. Log**
Append an entry to `experiments/loop-YYYYMMDD/log.md` regardless of outcome. Required fields: date, goal (one-sentence hypothesis), implementation summary, results table with per-video `mean_error_px` + `HOTA` and deltas vs previous best, conclusion. Use sub-entries (Xa, Xb, …) for iterative tuning within the same idea.

**7. Commit**
Commit all changes to the worktree branch with a descriptive message. The committed state must always leave `config.yaml` reflecting the current best results.

**8. Loop**
Return to step 1 immediately. Do not pause or ask for confirmation between iterations.

---

### Tracking Targets

When the focus area is tracking, maintain **two** best states simultaneously:

| Target | Criterion |
|--------|-----------|
| **Accuracy best** | Lowest aggregate mean error / highest HOTA, regardless of processing time |
| **Speed best** | Reaches ≥ 95% of accuracy-best performance with meaningfully lower processing time |

A change may advance one target without advancing the other — both are worth pursuing independently.

### Architecture Criterion

When choosing between approaches of similar expected performance, prefer the one that:
- Adds a single isolated processing step over one that modifies multiple interdependent components.
- Can be independently disabled or tuned via `config.yaml` without touching other stages.
- Does not increase coupling between pipeline stages.
