# Project Guidelines — AI Judging

Markerless 3D pose estimation pipeline for freeride skiing competition judging. Converts monocular video → 3D skeletal model with ski detection.

## Code Style

- **Python 3.12** (strict: `>=3.12, <3.13`). Use `from __future__ import annotations` in every module.
- Type hints throughout: use PEP 604 unions (`str | None`, not `Optional[str]`).
- **Ruff** for linting/formatting (`uv run ruff check src/`).
- Google/NumPy-style docstrings with `Parameters` / `Returns` sections.
- Naming: `snake_case` functions (verb-first: `extract_frames`, `detect_skis`), `_prefixed` private helpers, `UPPER_SNAKE_CASE` constants.
- Absolute imports only: `from src.segmentation.yolo_seg import segment_skier` (no relative imports).

## Architecture

Six-stage disk-coupled pipeline orchestrated by [src/pipeline.py](src/pipeline.py). Stages communicate via **JSON manifests** and image directories under `output/<stage>/<video_stem>/`.

Each stage is a single-module package with one public function following: `fn(input_path, output_dir, *, config_kwargs...) -> Path`. See [src/segmentation/yolo_seg.py](src/segmentation/yolo_seg.py) as the canonical example.

**Stages**: Frame Extraction → Person Segmentation (YOLO) → 2D Pose (YOLO) → 3D Lifting (MotionBERT) → Ski Detection (GroundingDINO+SAM2 / colour fallback) → Visualization (Plotly 3D / OpenCV).

**Import fallback pattern** (mandatory for ML-model stages): wrap heavy imports in `try/except ImportError` and fall back to a `_write_stub_*()` function. This lets the pipeline run end-to-end without all models installed. See [src/segmentation/yolo_seg.py](src/segmentation/yolo_seg.py) and [src/ski_detection/detector.py](src/ski_detection/detector.py).

Config is a plain `dict` loaded from [config.yaml](config.yaml) — accessed via `config.get("key", default)`. No Pydantic/dataclass schema.

## Build and Test

```bash
# Install core deps
uv sync

# Run pipeline
uv run python -m src.pipeline raw_videos/VIDEO.mp4
uv run python -m src.pipeline --test VIDEO.mp4   # quick: 5fps/50 frames
uv run python -m src.pipeline --max-videos 3     # batch

# Tests & lint
uv run pytest
uv run ruff check src/
```

## Running Individual Pipeline Stages

Use `--stage` to run a single stage independently, reusing upstream outputs. Useful for performance comparison with different hyperparameters.

**Available stages**: `frames`, `segmentation`, `tracking`, `pose_2d`, `pose_3d`, `ski_detection`, `visualization`.

```bash
# Run only tracking stage (reuses segmentation output)
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage tracking

# Run 2D pose with different config
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage pose_2d

# Test frame extraction in quick mode
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage frames --test

# Regenerate visualizations
uv run python -m src.pipeline raw_videos/VIDEO.mp4 --stage visualization
```

Each stage validates that required upstream outputs exist and fails with a clear error if dependencies are missing (e.g., tracking requires segmentation output).

## Project Conventions

- **One public function per stage module**. Add new stages as `src/<stage_name>/<module>.py` with empty `__init__.py`.
- All stage outputs go to `output/<stage>/<video_stem>/`. Return `Path` to the primary output file.
- Default device is `mps` (Apple Silicon). Support `cuda:0` and `cpu` via config.
- Heavy dependencies (`torch`, `ultralytics`) are deferred or guarded by `try/except ImportError`.
- No custom exception classes — use `FileNotFoundError`, `RuntimeError`, `ImportError` with descriptive messages.

## Integration Points

- **Ultralytics YOLO**: segmentation & pose models loaded from local checkpoints.
- **MotionBERT / GroundingDINO+SAM2**: referenced but currently stub implementations (TODO).
- Inter-stage coupling: 2D pose reads `bbox_padded` offsets from segmentation/tracking JSON to remap crop-space keypoints to frame coordinates.
