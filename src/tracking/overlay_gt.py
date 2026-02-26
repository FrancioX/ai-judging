"""Render overlay videos comparing tracker predictions vs ground-truth centers.

Draws on each frame:
- **Green** circle + trace: ground-truth (interpolated) center
- **Red** circle + trace: tracker-predicted center
- **Yellow** line: error vector between them
- **HUD**: per-frame error in pixels and running mean

Produces an MP4 video in ``output/tracking/<video_stem>/overlay_gt.mp4``.

Usage
-----
::

    # Single video
    uv run python -m src.tracking.overlay_gt \\
        "output/tracking/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv"

    # All annotated videos
    uv run python -m src.tracking.overlay_gt --batch
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.tracking.evaluate import (
    _ANNOTATIONS_DIR,
    _TRACKING_DIR,
    _interpolate_gt,
    _load_center_gt,
    _load_tracking_centers,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FRAMES_DIR = _PROJECT_ROOT / "output" / "frames"

# Trace trail length (number of past frames to draw)
_TRACE_LENGTH = 60

# Colours (BGR)
_GREEN = (0, 220, 0)
_RED = (0, 0, 220)
_YELLOW = (0, 220, 220)
_WHITE = (255, 255, 255)
_BG = (0, 0, 0)


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def render_overlay(
    tracking_dir: Path,
    gt_path: Path,
    *,
    output_path: Path | None = None,
    fps: float | None = None,
    trace_length: int = _TRACE_LENGTH,
) -> Path:
    """Render an overlay video with GT and predicted centers.

    Parameters
    ----------
    tracking_dir : Path
        Path to ``output/tracking/<video_stem>/``.
    gt_path : Path
        Path to ground-truth center CSV.
    output_path : Path | None
        Output MP4 path. Defaults to ``<tracking_dir>/overlay_gt.mp4``.
    fps : float | None
        Output video FPS. Auto-detected from ``frames_meta.json`` if None.
    trace_length : int
        Number of past frames to include in the trailing trace line.

    Returns
    -------
    Path
        Path to the rendered MP4 video.
    """
    video_stem = tracking_dir.name

    # Load data
    gt_sparse = _load_center_gt(gt_path)
    preds = _load_tracking_centers(tracking_dir)
    all_pred_ids = sorted(preds.keys())
    gt_dense = _interpolate_gt(gt_sparse, all_pred_ids)

    # Discover frames
    frames_dir = _FRAMES_DIR / video_stem
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # Auto-detect FPS
    if fps is None:
        meta_path = frames_dir / "frames_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            fps = meta.get("effective_fps", meta.get("source_fps", 25.0))
        else:
            fps = 25.0

    # Output path
    if output_path is None:
        output_path = tracking_dir / "overlay_gt.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read first frame to get dimensions
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Trailing traces
    gt_trace: deque[tuple[int, int]] = deque(maxlen=trace_length)
    pred_trace: deque[tuple[int, int]] = deque(maxlen=trace_length)

    # Running error stats
    errors: list[float] = []

    # Frame-id extractor
    def _fid(p: Path) -> int:
        return int(p.stem.split("_")[-1])

    gt_keyframe_set = set(gt_sparse.keys())

    print(f"\nRendering overlay: {video_stem}")
    print(f"  Frames: {len(frame_files)}, FPS: {fps}, GT keyframes: {len(gt_sparse)}")

    for frame_path in tqdm(frame_files, desc="  Rendering", unit="frame"):
        fid = _fid(frame_path)
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        has_gt = fid in gt_dense
        has_pred = fid in preds

        gt_pt = gt_dense.get(fid)
        pred_pt = preds.get(fid)

        # Compute per-frame error
        frame_error = None
        if gt_pt and pred_pt:
            frame_error = np.sqrt((gt_pt[0] - pred_pt[0]) ** 2 + (gt_pt[1] - pred_pt[1]) ** 2)
            errors.append(frame_error)

        # Update traces
        if gt_pt:
            gt_trace.append((int(gt_pt[0]), int(gt_pt[1])))
        if pred_pt:
            pred_trace.append((int(pred_pt[0]), int(pred_pt[1])))

        # --- Draw traces (fading trails) ---
        _draw_trace(img, gt_trace, _GREEN)
        _draw_trace(img, pred_trace, _RED)

        # --- Draw error line ---
        if gt_pt and pred_pt:
            cv2.line(
                img,
                (int(gt_pt[0]), int(gt_pt[1])),
                (int(pred_pt[0]), int(pred_pt[1])),
                _YELLOW, 1, cv2.LINE_AA,
            )

        # --- Draw current points ---
        if gt_pt:
            gx, gy = int(gt_pt[0]), int(gt_pt[1])
            cv2.circle(img, (gx, gy), 8, _GREEN, 2, cv2.LINE_AA)
            cv2.circle(img, (gx, gy), 2, _GREEN, -1, cv2.LINE_AA)
            # Mark GT keyframes with a larger ring
            if fid in gt_keyframe_set:
                cv2.circle(img, (gx, gy), 14, _GREEN, 1, cv2.LINE_AA)

        if pred_pt:
            px, py = int(pred_pt[0]), int(pred_pt[1])
            cv2.circle(img, (px, py), 8, _RED, 2, cv2.LINE_AA)
            cv2.circle(img, (px, py), 2, _RED, -1, cv2.LINE_AA)

        # --- HUD ---
        _draw_hud(img, fid, frame_error, errors, has_gt, has_pred, w)

        writer.write(img)

    writer.release()
    print(f"  Saved → {output_path}  ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def _draw_trace(
    img: np.ndarray,
    trace: deque[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    """Draw a fading polyline trace on the image.

    Parameters
    ----------
    img : np.ndarray
        Image to draw on (modified in-place).
    trace : deque[tuple[int, int]]
        Recent center positions.
    color : tuple[int, int, int]
        BGR colour for the trace.
    """
    pts = list(trace)
    n = len(pts)
    if n < 2:
        return
    for i in range(1, n):
        alpha = i / n  # 0 = oldest (faint), 1 = newest (bright)
        thickness = max(1, int(alpha * 3))
        faded = tuple(int(c * alpha * 0.7) for c in color)
        cv2.line(img, pts[i - 1], pts[i], faded, thickness, cv2.LINE_AA)


def _draw_hud(
    img: np.ndarray,
    frame_id: int,
    frame_error: float | None,
    errors: list[float],
    has_gt: bool,
    has_pred: bool,
    img_width: int,
) -> None:
    """Draw a heads-up display with error stats.

    Parameters
    ----------
    img : np.ndarray
        Image to draw on (modified in-place).
    frame_id : int
        Current frame number.
    frame_error : float | None
        Pixel error for this frame (None if GT or pred missing).
    errors : list[float]
        All errors so far (for running mean).
    has_gt : bool
        Whether GT exists for this frame.
    has_pred : bool
        Whether a prediction exists for this frame.
    img_width : int
        Image width (for positioning).
    """
    # Background bar
    cv2.rectangle(img, (0, 0), (img_width, 70), _BG, -1)

    # Legend
    cv2.circle(img, (15, 18), 6, _GREEN, -1)
    cv2.putText(img, "GT", (28, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 1, cv2.LINE_AA)
    cv2.circle(img, (70, 18), 6, _RED, -1)
    cv2.putText(img, "Pred", (83, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _RED, 1, cv2.LINE_AA)

    # Frame number
    cv2.putText(
        img, f"Frame {frame_id}", (140, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA,
    )

    # Error for this frame
    if frame_error is not None:
        err_color = _GREEN if frame_error < 30 else _YELLOW if frame_error < 60 else _RED
        cv2.putText(
            img, f"Error: {frame_error:.1f} px", (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, err_color, 2, cv2.LINE_AA,
        )
    elif not has_gt:
        cv2.putText(
            img, "No GT", (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 1, cv2.LINE_AA,
        )

    # Running mean
    if errors:
        mean_err = np.mean(errors)
        cv2.putText(
            img, f"Mean: {mean_err:.1f} px", (250, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1, cv2.LINE_AA,
        )

    # Error bar visualization (right side)
    if frame_error is not None:
        bar_max = 100.0
        bar_width = min(int(frame_error / bar_max * 200), 200)
        bar_x = img_width - 220
        bar_color = _GREEN if frame_error < 30 else _YELLOW if frame_error < 60 else _RED
        cv2.rectangle(img, (bar_x, 38), (bar_x + bar_width, 58), bar_color, -1)
        cv2.rectangle(img, (bar_x, 38), (bar_x + 200, 58), (80, 80, 80), 1)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def render_all(
    annotations_dir: Path | None = None,
    tracking_dir: Path | None = None,
    **kwargs: float | int,
) -> list[Path]:
    """Render overlay videos for all annotated videos.

    Parameters
    ----------
    annotations_dir : Path | None
        Root annotations dir (default ``annotations/tracking/``).
    tracking_dir : Path | None
        Root tracking dir (default ``output/tracking/``).
    **kwargs
        Forwarded to :func:`render_overlay`.

    Returns
    -------
    list[Path]
        Paths to rendered MP4 files.
    """
    ann_root = annotations_dir or _ANNOTATIONS_DIR
    trk_root = tracking_dir or _TRACKING_DIR

    if not ann_root.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ann_root}")

    outputs = []
    for gt_csv in sorted(ann_root.rglob("gt_centers.csv")):
        video_stem = gt_csv.parent.name
        trk_dir = trk_root / video_stem
        if not trk_dir.exists():
            print(f"  SKIP {video_stem} — no tracking output")
            continue
        out = render_overlay(trk_dir, gt_csv, **kwargs)
        outputs.append(out)

    print(f"\nRendered {len(outputs)} overlay video(s).")
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    if "--batch" in sys.argv:
        render_all()
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage: uv run python -m src.tracking.overlay_gt <tracking_dir> <gt_csv>")
        print("       uv run python -m src.tracking.overlay_gt --batch")
        sys.exit(1)

    render_overlay(Path(sys.argv[1]), Path(sys.argv[2]))
