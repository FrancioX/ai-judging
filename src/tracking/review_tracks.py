"""Track-seeded center-point review tool using OpenCV.

Opens extracted frames with tracker-predicted centers pre-populated. The user
only needs to interact when the tracker is wrong. Accepted frames are saved as
ground-truth annotations in the same CSV format used by ``annotate_centers.py``.

Controls
--------
- **SPACE / RIGHT**: Accept predicted center for this frame and advance
- **LEFT**: Go back one frame
- **Left-click**: Override center with click position (marks frame as CORRECTED)
- **u**: Undo correction — revert to auto-predicted center
- **r**: Remove this frame from annotations (exclude from GT)
- **a**: Accept all remaining frames as-is, save and quit
- **j / k**: Jump forward / backward 10 frames
- **q**: Save and quit

Usage
-----
::

    uv run python -m src.tracking.review_tracks \\
        "output/tracking/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv" \\
        [--step=N]       # review every N-th frame (default: 1)
        [--track-id=N]   # track ID to write in CSV (default: 1)
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

def _load_tracking(tracking_dir: Path) -> tuple[dict[int, tuple[float, float]], dict[int, list[int]], dict[int, bool]]:
    """Load per-frame data from tracking.json.

    Returns
    -------
    tuple of:
        auto_centers : dict[frame_id, (cx, cy)]  — bbox centers from tracker
        bboxes       : dict[frame_id, [x1,y1,x2,y2]]
        interpolated : dict[frame_id, bool]       — True when tracker interpolated
    """
    manifest = tracking_dir / "tracking.json"
    if not manifest.exists():
        raise FileNotFoundError(f"tracking.json not found in {tracking_dir}")

    data = json.loads(manifest.read_text())
    auto_centers: dict[int, tuple[float, float]] = {}
    bboxes: dict[int, list[int]] = {}
    interpolated: dict[int, bool] = {}

    for frame in data.get("frames", []):
        fid = frame["frame_id"]
        bbox = frame.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            auto_centers[fid] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            bboxes[fid] = bbox
            interpolated[fid] = not frame.get("detected", True)

    return auto_centers, bboxes, interpolated


def _save_annotations(
    output_csv: Path,
    annotations: dict[int, tuple[float, float]],
    track_id: int,
) -> None:
    """Write annotations to CSV (same format as annotate_centers.py)."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        f.write("# frame_id,track_id,center_x,center_y\n")
        writer = csv.writer(f)
        for fid in sorted(annotations):
            cx, cy = annotations[fid]
            writer.writerow([fid, track_id, f"{cx:.1f}", f"{cy:.1f}"])
    print(f"\nSaved {len(annotations)} annotations → {output_csv}")


def _draw_center(
    img,
    cx: float,
    cy: float,
    color: tuple[int, int, int],
    radius: int = 10,
) -> None:
    icx, icy = int(cx), int(cy)
    cv2.circle(img, (icx, icy), radius, color, 2)
    cv2.circle(img, (icx, icy), 2, color, -1)
    cv2.line(img, (icx - 15, icy), (icx + 15, icy), color, 1)
    cv2.line(img, (icx, icy - 15), (icx, icy + 15), color, 1)


def _draw_bbox_overlay(img, bbox: list[int], alpha: float = 0.25) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 50), 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def review_tracks(
    tracking_dir: Path,
    output_csv: Path,
    *,
    track_id: int = 1,
    frame_step: int = 1,
) -> Path:
    """Open an interactive window to review and correct tracker-predicted centers.

    Parameters
    ----------
    tracking_dir : Path
        Directory containing ``tracking.json`` (e.g. ``output/tracking/<stem>``).
    output_csv : Path
        Where to write the annotation CSV.
    track_id : int
        Track ID to assign in CSV (default ``1``).
    frame_step : int
        Show every N-th frame. Intermediate frames pre-populated from tracker
        are still written to the CSV unless ``--step`` is used to limit scope.

    Returns
    -------
    Path
        Path to the saved annotation CSV.
    """
    global _click_pos

    tracking_dir = Path(tracking_dir)
    output_csv = Path(output_csv)

    # Derive frames directory from tracking dir path
    frames_dir = tracking_dir.parent.parent / "frames" / tracking_dir.name
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Load tracker predictions
    auto_centers, bboxes, interpolated_flags = _load_tracking(tracking_dir)
    print(f"Loaded {len(auto_centers)} predicted centers from tracking.json")

    # Discover frame files
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    def _frame_id(p: Path) -> int:
        return int(p.stem.split("_")[-1])

    # Build display list (every step-th frame that has a tracker prediction)
    all_display = list(range(0, len(frame_files), frame_step))
    display_indices = [i for i in all_display
                       if _frame_id(frame_files[i]) in auto_centers]
    total_display = len(display_indices)

    # Pre-populate annotations from tracker (only frames we will display)
    # Resuming: load existing CSV if present
    annotations: dict[int, tuple[float, float]] = {}
    corrected_frames: set[int] = set()

    if output_csv.exists():
        with open(output_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                fid = int(row[0])
                annotations[fid] = (float(row[2]), float(row[3]))
        print(f"Resumed {len(annotations)} existing annotations from {output_csv}")
        # Mark resumed annotations that differ from auto as corrected
        for fid, pos in annotations.items():
            ac = auto_centers.get(fid)
            if ac is None or abs(pos[0] - ac[0]) > 0.5 or abs(pos[1] - ac[1]) > 0.5:
                corrected_frames.add(fid)
    else:
        # Seed all displayed frames from tracker
        for i in display_indices:
            fid = _frame_id(frame_files[i])
            annotations[fid] = auto_centers[fid]

    print(f"\nReview session")
    print(f"  Tracking dir : {tracking_dir}")
    print(f"  Frames dir   : {frames_dir}")
    print(f"  Total frames : {len(frame_files)}  |  Showing every {frame_step}th → {total_display} frames")
    print(f"  Output       : {output_csv}")
    print(f"\nColors: CYAN=auto  GREEN=corrected  YELLOW=interpolated (less reliable)")
    print(f"Controls: SPACE/→=accept  ←=back  click=correct  u=undo  r=remove  a=accept-all  j/k=±10  q=save+quit\n")

    cv2.namedWindow("Review Tracks", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Review Tracks", 1280, 720)
    cv2.setMouseCallback("Review Tracks", _mouse_callback)

    idx = 0
    while 0 <= idx < total_display:
        fi = display_indices[idx]
        frame_path = frame_files[fi]
        frame_id = _frame_id(frame_path)

        img = cv2.imread(str(frame_path))
        if img is None:
            idx += 1
            continue

        def _render(image, fid: int) -> object:
            display = image.copy()

            # Draw bbox overlay
            if fid in bboxes:
                _draw_bbox_overlay(display, bboxes[fid])

            # Draw center
            if fid in annotations:
                cx, cy = annotations[fid]
                is_interp = interpolated_flags.get(fid, False)
                is_corrected = fid in corrected_frames
                if is_corrected:
                    color = (0, 255, 0)      # green
                    label = "CORRECTED"
                elif is_interp:
                    color = (0, 220, 255)    # yellow
                    label = "AUTO (interpolated)"
                else:
                    color = (255, 220, 0)    # cyan
                    label = "AUTO"
                _draw_center(display, cx, cy, color)
            else:
                label = "REMOVED"

            bar = (
                f"Frame {fid}  [{idx + 1}/{total_display}]  "
                f"Corrected: {len(corrected_frames)}  |  {label}"
            )
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
            bar_color = (0, 255, 0) if fid in corrected_frames else (0, 255, 255)
            cv2.putText(display, bar, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
            return display

        display = _render(img, frame_id)
        cv2.imshow("Review Tracks", display)
        _click_pos = None

        while True:
            key = cv2.waitKey(30) & 0xFF

            # Handle click → override center
            if _click_pos is not None:
                cx, cy = _click_pos
                annotations[frame_id] = (float(cx), float(cy))
                corrected_frames.add(frame_id)
                _click_pos = None
                display = _render(img, frame_id)
                cv2.imshow("Review Tracks", display)

            # SPACE or RIGHT → accept and advance
            if key == ord(" ") or key == 83 or key == 3:
                idx += 1
                break

            # LEFT → go back
            if key == 81 or key == 2:
                idx = max(0, idx - 1)
                break

            # 'u' → undo correction, revert to auto
            if key == ord("u"):
                corrected_frames.discard(frame_id)
                ac = auto_centers.get(frame_id)
                if ac:
                    annotations[frame_id] = ac
                else:
                    annotations.pop(frame_id, None)
                display = _render(img, frame_id)
                cv2.imshow("Review Tracks", display)

            # 'r' → remove this frame from annotations
            if key == ord("r"):
                annotations.pop(frame_id, None)
                corrected_frames.discard(frame_id)
                display = _render(img, frame_id)
                cv2.imshow("Review Tracks", display)

            # 'j' → jump +10
            if key == ord("j"):
                idx = min(total_display - 1, idx + 10)
                break

            # 'k' → jump -10
            if key == ord("k"):
                idx = max(0, idx - 10)
                break

            # 'a' → accept all remaining and quit
            if key == ord("a"):
                # All remaining frames are already pre-populated; just quit
                idx = total_display
                break

            # 'q' → save and quit
            if key == ord("q"):
                idx = total_display
                break

    cv2.destroyAllWindows()

    _save_annotations(output_csv, annotations, track_id)
    return output_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: uv run python -m src.tracking.review_tracks <tracking_dir> <output_csv> [options]")
        print()
        print("Options:")
        print("  --step=N        Review every N-th frame (default: 1)")
        print("  --track-id=N    Track ID written to CSV (default: 1)")
        sys.exit(1)

    tracking_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    kwargs: dict = {}
    for arg in sys.argv[3:]:
        if arg.startswith("--step="):
            kwargs["frame_step"] = int(arg.split("=")[1])
        elif arg.startswith("--track-id="):
            kwargs["track_id"] = int(arg.split("=")[1])

    review_tracks(tracking_dir, output_csv, **kwargs)
