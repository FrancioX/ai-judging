"""Lightweight center-point annotation tool using OpenCV.

Opens extracted frames in a window. Click the skier's approximate center
in each displayed frame.

Controls
--------
- **Left-click**: mark skier center (overrides tracking suggestion)
- **SPACE / RIGHT**: next frame (auto-accepts tracking suggestion if no manual annotation)
- **LEFT**: previous frame
- **s**: skip frame (no annotation)
- **j / k**: jump forward / backward 10 frames
- **u**: undo annotation on current frame
- **q**: save and quit

Tracking-assisted mode
-----------------------
Pass ``--tracking=<path_to_tracking.json>`` to pre-populate suggestions from
tracking output. Suggestions are shown in **cyan**. Press SPACE to accept the
suggestion, or click to override it.

Sparse annotations (e.g. every 10th frame via ``--step``) are linearly
interpolated during evaluation, so you don't need every frame.

Usage
-----
::

    uv run python -m src.tracking.annotate_centers \\
        "output/frames/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv" \\
        --step=10

    # With tracking suggestions:
    uv run python -m src.tracking.annotate_centers \\
        "output/frames/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv" \\
        --step=10 \\
        --tracking="output/tracking/<video_stem>/tracking.json"
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Module-level state for mouse callback
# ---------------------------------------------------------------------------
_click_pos: tuple[int, int] | None = None


def _mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    """Record left-click position."""
    global _click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_pos = (x, y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tracking_hints(tracking_json: Path) -> dict[int, tuple[float, float]]:
    """Load per-frame center hints from a tracking manifest.

    Parameters
    ----------
    tracking_json : Path
        Path to ``tracking.json`` produced by :mod:`src.tracking.tracker`.

    Returns
    -------
    dict[int, tuple[float, float]]
        Mapping ``frame_id → (center_x, center_y)`` derived from bbox midpoints.
    """
    with open(tracking_json) as f:
        data = json.load(f)

    hints: dict[int, tuple[float, float]] = {}
    for entry in data.get("frames", []):
        fid = entry.get("frame_id")
        bbox = entry.get("bbox")  # [x1, y1, x2, y2]
        if fid is not None and bbox and len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            hints[fid] = (cx, cy)
    return hints


def _draw_hint(display, cx: float, cy: float) -> None:
    """Draw a cyan tracking suggestion marker."""
    icx, icy = int(cx), int(cy)
    cv2.circle(display, (icx, icy), 12, (255, 200, 0), 2)
    cv2.circle(display, (icx, icy), 3, (255, 200, 0), -1)
    cv2.line(display, (icx - 18, icy), (icx + 18, icy), (255, 200, 0), 1)
    cv2.line(display, (icx, icy - 18), (icx, icy + 18), (255, 200, 0), 1)
    cv2.putText(
        display, "TRACK",
        (icx + 14, icy - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1,
    )


def _draw_annotation(display, cx: float, cy: float, accepted_from_track: bool = False) -> None:
    """Draw a confirmed annotation marker (green, or teal if auto-accepted)."""
    color = (0, 200, 100) if accepted_from_track else (0, 255, 0)
    icx, icy = int(cx), int(cy)
    cv2.circle(display, (icx, icy), 10, color, 2)
    cv2.circle(display, (icx, icy), 2, color, -1)
    cv2.line(display, (icx - 15, icy), (icx + 15, icy), color, 1)
    cv2.line(display, (icx, icy - 15), (icx, icy + 15), color, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_video(
    frames_dir: Path,
    output_csv: Path,
    *,
    track_id: int = 1,
    frame_step: int = 1,
    resume_csv: Path | None = None,
    tracking_hints: dict[int, tuple[float, float]] | None = None,
) -> Path:
    """Open an interactive window to click-annotate skier center points.

    Parameters
    ----------
    frames_dir : Path
        Directory containing extracted frames (``frame_NNNNNN.jpg``).
    output_csv : Path
        Where to write the annotation CSV.
    track_id : int
        Track ID to assign (default ``1`` for single-skier videos).
    frame_step : int
        Show every N-th frame for annotation. Intermediate frames are
        filled by linear interpolation during evaluation.
    resume_csv : Path | None
        Path to previously saved CSV to resume annotation from.
    tracking_hints : dict[int, tuple[float, float]] | None
        Optional pre-computed tracking center points keyed by frame_id.
        Shown in cyan; pressing SPACE auto-accepts the suggestion.

    Returns
    -------
    Path
        Path to the saved annotation CSV.
    """
    global _click_pos

    # Discover frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # Parse frame_id from filename (e.g. frame_000150.jpg → 150)
    def _frame_id(p: Path) -> int:
        return int(p.stem.split("_")[-1])

    # Load existing annotations if resuming
    annotations: dict[int, tuple[float, float]] = {}
    # Track which annotations were auto-accepted from tracking (for display color)
    auto_accepted: set[int] = set()

    if resume_csv and resume_csv.exists():
        with open(resume_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                fid = int(row[0])
                annotations[fid] = (float(row[2]), float(row[3]))
        print(f"Resumed {len(annotations)} existing annotations from {resume_csv}")

    has_hints = bool(tracking_hints)
    hints = tracking_hints or {}

    # Build display list (every step-th frame)
    display_indices = list(range(0, len(frame_files), frame_step))
    total_display = len(display_indices)

    print(f"\nAnnotation session")
    print(f"  Frames dir : {frames_dir}")
    print(f"  Total frames: {len(frame_files)}")
    print(f"  Showing every {frame_step}th → {total_display} frames to annotate")
    print(f"  Output     : {output_csv}")
    if has_hints:
        hint_coverage = sum(1 for i in display_indices if _frame_id(frame_files[i]) in hints)
        print(f"  Tracking hints: {len(hints)} frames ({hint_coverage}/{total_display} displayed frames covered)")
        print(f"  Cyan dot = tracking suggestion  |  SPACE = accept suggestion")
    print(f"\nControls: click=mark  SPACE/→=next  ←=prev  s=skip  j/k=±10  u=undo  q=save+quit\n")

    cv2.namedWindow("Annotate Skier Center", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate Skier Center", 1280, 720)
    cv2.setMouseCallback("Annotate Skier Center", _mouse_callback)

    def _render_frame(img, frame_id: int, idx: int, status_msg: str, status_color=(0, 255, 255)):
        """Build display frame with overlays and status bar."""
        display = img.copy()
        hint = hints.get(frame_id)

        # Draw tracking hint (cyan) if present and frame not yet annotated
        if hint and frame_id not in annotations:
            _draw_hint(display, *hint)

        # Draw confirmed annotation
        if frame_id in annotations:
            cx, cy = annotations[frame_id]
            _draw_annotation(display, cx, cy, accepted_from_track=(frame_id in auto_accepted))
            # Still show hint in background if it differs significantly from annotation
            if hint:
                hx, hy = hint
                if abs(hx - cx) > 5 or abs(hy - cy) > 5:
                    _draw_hint(display, hx, hy)

        ann_status = "ANNOTATED" if frame_id in annotations else ("HINT" if hint else "---")
        bar = (
            f"Frame {frame_id}  [{idx + 1}/{total_display}]  "
            f"Annotated: {len(annotations)}  |  {ann_status}  |  {status_msg}"
        )
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(display, bar, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        return display

    idx = 0
    while 0 <= idx < total_display:
        fi = display_indices[idx]
        frame_path = frame_files[fi]
        frame_id = _frame_id(frame_path)

        img = cv2.imread(str(frame_path))
        if img is None:
            idx += 1
            continue

        hint = hints.get(frame_id)
        display = _render_frame(img, frame_id, idx, "")
        cv2.imshow("Annotate Skier Center", display)
        _click_pos = None

        while True:
            key = cv2.waitKey(30) & 0xFF

            # Handle click → manual annotation
            if _click_pos is not None:
                cx, cy = _click_pos
                annotations[frame_id] = (float(cx), float(cy))
                auto_accepted.discard(frame_id)
                _click_pos = None
                display = _render_frame(img, frame_id, idx, f"MARKED ({cx:.0f}, {cy:.0f})", (0, 255, 0))
                cv2.imshow("Annotate Skier Center", display)

            # SPACE or RIGHT → next (auto-accept hint if no annotation)
            if key == ord(" ") or key == 83 or key == 3:
                if frame_id not in annotations and hint:
                    annotations[frame_id] = hint
                    auto_accepted.add(frame_id)
                idx += 1
                break
            # LEFT → previous
            if key == 81 or key == 2:
                idx = max(0, idx - 1)
                break
            # 's' → skip (no annotation)
            if key == ord("s"):
                idx += 1
                break
            # 'j' → jump +10
            if key == ord("j"):
                idx = min(total_display - 1, idx + 10)
                break
            # 'k' → jump -10
            if key == ord("k"):
                idx = max(0, idx - 10)
                break
            # 'u' → undo current frame
            if key == ord("u"):
                annotations.pop(frame_id, None)
                auto_accepted.discard(frame_id)
                display = _render_frame(img, frame_id, idx, "UNDONE", (0, 0, 255))
                cv2.imshow("Annotate Skier Center", display)
            # 'q' → save and quit
            if key == ord("q"):
                idx = total_display  # exit outer loop
                break

    cv2.destroyAllWindows()

    # Save
    _save_annotations(output_csv, annotations, track_id)
    return output_csv


def _save_annotations(
    output_csv: Path,
    annotations: dict[int, tuple[float, float]],
    track_id: int,
) -> None:
    """Write annotations to CSV.

    Parameters
    ----------
    output_csv : Path
        Destination file.
    annotations : dict[int, tuple[float, float]]
        ``{frame_id: (center_x, center_y)}``.
    track_id : int
        Track identity to write.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        f.write("# frame_id,track_id,center_x,center_y\n")
        writer = csv.writer(f)
        for fid in sorted(annotations):
            cx, cy = annotations[fid]
            writer.writerow([fid, track_id, f"{cx:.1f}", f"{cy:.1f}"])
    print(f"\nSaved {len(annotations)} annotations → {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: uv run python -m src.tracking.annotate_centers <frames_dir> <output_csv> [options]")
        print()
        print("Options:")
        print("  --step=N              Annotate every N-th frame (default: 10)")
        print("  --resume=<csv>        Resume from a previous annotation CSV")
        print("  --tracking=<json>     Load tracking suggestions from tracking.json")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    kwargs: dict = {}
    for arg in sys.argv[3:]:
        if arg.startswith("--step="):
            kwargs["frame_step"] = int(arg.split("=")[1])
        elif arg.startswith("--resume="):
            kwargs["resume_csv"] = Path(arg.split("=", 1)[1])
        elif arg.startswith("--tracking="):
            tracking_path = Path(arg.split("=", 1)[1])
            if not tracking_path.exists():
                print(f"Warning: tracking file not found: {tracking_path}")
            else:
                kwargs["tracking_hints"] = _load_tracking_hints(tracking_path)

    annotate_video(frames_dir, output_csv, **kwargs)
