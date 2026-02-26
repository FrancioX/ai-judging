"""Lightweight center-point annotation tool using OpenCV.

Opens extracted frames in a window. Click the skier's approximate center
in each displayed frame.

Controls
--------
- **Left-click**: mark skier center
- **SPACE / RIGHT**: next frame
- **LEFT**: previous frame
- **s**: skip frame (no annotation)
- **j / k**: jump forward / backward 10 frames
- **u**: undo annotation on current frame
- **q**: save and quit

Sparse annotations (e.g. every 10th frame via ``--step``) are linearly
interpolated during evaluation, so you don't need every frame.

Usage
-----
::

    uv run python -m src.tracking.annotate_centers \\
        "output/frames/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv" \\
        --step=10
"""

from __future__ import annotations

import csv
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
# Public API
# ---------------------------------------------------------------------------

def annotate_video(
    frames_dir: Path,
    output_csv: Path,
    *,
    track_id: int = 1,
    frame_step: int = 1,
    resume_csv: Path | None = None,
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
    if resume_csv and resume_csv.exists():
        with open(resume_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                fid = int(row[0])
                annotations[fid] = (float(row[2]), float(row[3]))
        print(f"Resumed {len(annotations)} existing annotations from {resume_csv}")

    # Build display list (every step-th frame)
    display_indices = list(range(0, len(frame_files), frame_step))
    total_display = len(display_indices)

    print(f"\nAnnotation session")
    print(f"  Frames dir : {frames_dir}")
    print(f"  Total frames: {len(frame_files)}")
    print(f"  Showing every {frame_step}th → {total_display} frames to annotate")
    print(f"  Output     : {output_csv}")
    print(f"\nControls: click=mark  SPACE/→=next  ←=prev  s=skip  j/k=±10  u=undo  q=save+quit\n")

    cv2.namedWindow("Annotate Skier Center", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate Skier Center", 1280, 720)
    cv2.setMouseCallback("Annotate Skier Center", _mouse_callback)

    idx = 0
    while 0 <= idx < total_display:
        fi = display_indices[idx]
        frame_path = frame_files[fi]
        frame_id = _frame_id(frame_path)

        img = cv2.imread(str(frame_path))
        if img is None:
            idx += 1
            continue

        display = img.copy()

        # Draw existing annotation (green circle + crosshair)
        if frame_id in annotations:
            cx, cy = annotations[frame_id]
            icx, icy = int(cx), int(cy)
            cv2.circle(display, (icx, icy), 10, (0, 255, 0), 2)
            cv2.circle(display, (icx, icy), 2, (0, 255, 0), -1)
            cv2.line(display, (icx - 15, icy), (icx + 15, icy), (0, 255, 0), 1)
            cv2.line(display, (icx, icy - 15), (icx, icy + 15), (0, 255, 0), 1)

        # Status bar
        ann_status = "ANNOTATED" if frame_id in annotations else "---"
        bar = (
            f"Frame {frame_id}  [{idx + 1}/{total_display}]  "
            f"Total annotated: {len(annotations)}  |  {ann_status}"
        )
        # Dark background for text
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(display, bar, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Annotate Skier Center", display)
        _click_pos = None

        while True:
            key = cv2.waitKey(30) & 0xFF

            # Handle click
            if _click_pos is not None:
                cx, cy = _click_pos
                annotations[frame_id] = (float(cx), float(cy))
                _click_pos = None

                # Redraw with new annotation
                display = img.copy()
                icx, icy = int(cx), int(cy)
                cv2.circle(display, (icx, icy), 10, (0, 255, 0), 2)
                cv2.circle(display, (icx, icy), 2, (0, 255, 0), -1)
                cv2.line(display, (icx - 15, icy), (icx + 15, icy), (0, 255, 0), 1)
                cv2.line(display, (icx, icy - 15), (icx, icy + 15), (0, 255, 0), 1)

                bar = (
                    f"Frame {frame_id}  [{idx + 1}/{total_display}]  "
                    f"Total annotated: {len(annotations)}  |  MARKED ({cx:.0f}, {cy:.0f})"
                )
                cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(display, bar, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Annotate Skier Center", display)

            # SPACE or RIGHT → next
            if key == ord(" ") or key == 83 or key == 3:
                idx += 1
                break
            # LEFT → previous
            if key == 81 or key == 2:
                idx = max(0, idx - 1)
                break
            # 's' → skip
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
                # Redraw without annotation
                display = img.copy()
                bar = (
                    f"Frame {frame_id}  [{idx + 1}/{total_display}]  "
                    f"Total annotated: {len(annotations)}  |  UNDONE"
                )
                cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(display, bar, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
        print("  --step=N        Annotate every N-th frame (default: 10)")
        print("  --resume=<csv>  Resume from a previous annotation CSV")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    kwargs: dict = {}
    for arg in sys.argv[3:]:
        if arg.startswith("--step="):
            kwargs["frame_step"] = int(arg.split("=")[1])
        elif arg.startswith("--resume="):
            kwargs["resume_csv"] = Path(arg.split("=", 1)[1])

    annotate_video(frames_dir, output_csv, **kwargs)
