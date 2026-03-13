"""Center-point annotation tool using OpenCV.

Supports two modes:

1. **From-scratch** (default): Opens frames in a window. Click the skier's
   approximate center in each displayed frame. Optionally pass
   ``--tracking=<tracking.json>`` to show cyan hint markers.

2. **From-track** (``--from-track``): Pre-populates all frames from
   ``tracking.json``, shows bounding boxes and interpolation status. You only
   need to correct frames where the tracker is wrong.

Controls
--------
- **Left-click**: mark / override skier center
- **SPACE / RIGHT**: accept current annotation and advance
- **LEFT**: previous frame
- **s**: skip frame (no annotation)
- **j / k**: jump forward / backward 10 frames
- **u**: undo annotation on current frame (reverts to tracker prediction in
  from-track mode)
- **r**: remove frame from annotations
- **a**: accept all remaining frames as-is, save and quit (from-track mode)
- **q**: save and quit

Usage
-----
::

    # From scratch (click every frame)
    uv run python -m src.tracking.annotate_centers \\
        "output/frames/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv" \\
        --step=10

    # From scratch with tracking hints
    uv run python -m src.tracking.annotate_centers \\
        "output/frames/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv" \\
        --step=10 --tracking="output/tracking/<video_stem>/tracking.json"

    # Review / correct existing tracker output
    uv run python -m src.tracking.annotate_centers \\
        "output/frames/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_centers.csv" \\
        --from-track="output/tracking/<video_stem>"
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

def _load_tracking(source: Path) -> tuple[dict[int, tuple[float, float]], dict[int, list[int]], dict[int, bool]]:
    """Load per-frame data from tracking.json.

    Parameters
    ----------
    source : Path
        Either a directory containing ``tracking.json`` or the JSON file itself.

    Returns
    -------
    tuple of:
        centers      : dict[frame_id, (cx, cy)]
        bboxes       : dict[frame_id, [x1,y1,x2,y2]]
        interpolated : dict[frame_id, bool]
    """
    if source.is_dir():
        manifest = source / "tracking.json"
    else:
        manifest = source
    if not manifest.exists():
        raise FileNotFoundError(f"tracking.json not found at {manifest}")

    data = json.loads(manifest.read_text())
    centers: dict[int, tuple[float, float]] = {}
    bboxes: dict[int, list[int]] = {}
    interpolated: dict[int, bool] = {}

    for frame in data.get("frames", []):
        frame_file = frame.get("frame_file", "")
        try:
            fid = int(Path(frame_file).stem.split("_")[-1])
        except (ValueError, IndexError):
            fid = frame["frame_id"]
        bbox = frame.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            centers[fid] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            bboxes[fid] = bbox
            interpolated[fid] = not frame.get("detected", True)

    return centers, bboxes, interpolated


def _save_annotations(
    output_csv: Path,
    annotations: dict[int, tuple[float, float]],
    track_id: int,
) -> None:
    """Write annotations to CSV."""
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
    label: str | None = None,
) -> None:
    icx, icy = int(cx), int(cy)
    cv2.circle(img, (icx, icy), radius, color, 2)
    cv2.circle(img, (icx, icy), 2, color, -1)
    cv2.line(img, (icx - 15, icy), (icx + 15, icy), color, 1)
    cv2.line(img, (icx, icy - 15), (icx, icy + 15), color, 1)
    if label:
        cv2.putText(img, label, (icx + 14, icy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_bbox_overlay(img, bbox: list[int], alpha: float = 0.25) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 50), 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_video(
    frames_dir: Path,
    output_csv: Path,
    *,
    track_id: int = 1,
    frame_step: int = 1,
    from_track: Path | None = None,
    tracking_hints: dict[int, tuple[float, float]] | None = None,
) -> Path:
    """Open an interactive window to annotate skier center points.

    Parameters
    ----------
    frames_dir : Path
        Directory containing extracted frames (``frame_NNNNNN.jpg``).
    output_csv : Path
        Where to write the annotation CSV.
    track_id : int
        Track ID to assign (default ``1``).
    frame_step : int
        Show every N-th frame.
    from_track : Path | None
        Path to tracking directory or ``tracking.json``. When provided, all
        frames are pre-populated from tracker predictions (review mode).
    tracking_hints : dict[int, tuple[float, float]] | None
        Optional hint-only tracking centers (shown in cyan, not pre-populated).
        Ignored when *from_track* is set.

    Returns
    -------
    Path
        Path to the saved annotation CSV.
    """
    global _click_pos

    frames_dir = Path(frames_dir)
    output_csv = Path(output_csv)

    # Load tracking data
    auto_centers: dict[int, tuple[float, float]] = {}
    bboxes: dict[int, list[int]] = {}
    interpolated_flags: dict[int, bool] = {}
    hints: dict[int, tuple[float, float]] = {}
    seeded = False  # True when frames are pre-populated from tracking

    if from_track is not None:
        auto_centers, bboxes, interpolated_flags = _load_tracking(Path(from_track))
        hints = auto_centers
        seeded = True
        print(f"Loaded {len(auto_centers)} predicted centers from tracking")
    elif tracking_hints:
        hints = tracking_hints

    # Discover frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    def _frame_id(p: Path) -> int:
        return int(p.stem.split("_")[-1])

    # Build display list
    all_display = list(range(0, len(frame_files), frame_step))
    if seeded:
        # In from-track mode, only show frames that have tracker predictions
        display_indices = [i for i in all_display
                           if _frame_id(frame_files[i]) in auto_centers]
    else:
        display_indices = all_display
    total_display = len(display_indices)

    # Annotations and correction tracking
    annotations: dict[int, tuple[float, float]] = {}
    corrected_frames: set[int] = set()

    # Resume from existing CSV
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
    elif seeded:
        # Seed all displayed frames from tracker
        for i in display_indices:
            fid = _frame_id(frame_files[i])
            annotations[fid] = auto_centers[fid]

    # Print session info
    mode_label = "FROM-TRACK (review)" if seeded else "FROM-SCRATCH"
    print(f"\nAnnotation session — {mode_label}")
    print(f"  Frames dir : {frames_dir}")
    print(f"  Total frames: {len(frame_files)}  |  Showing every {frame_step}th → {total_display} frames")
    print(f"  Output     : {output_csv}")
    if hints and not seeded:
        hint_coverage = sum(1 for i in display_indices if _frame_id(frame_files[i]) in hints)
        print(f"  Tracking hints: {len(hints)} frames ({hint_coverage}/{total_display} displayed)")
    if seeded:
        print("\nColors: CYAN=auto  GREEN=corrected  YELLOW=interpolated (less reliable)")
    print(f"Controls: click=mark  SPACE/→=next  ←=prev  s=skip  j/k=±10  u=undo  r=remove  {'a=accept-all  ' if seeded else ''}q=save+quit\n")

    cv2.namedWindow("Annotate Centers", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate Centers", 1280, 720)
    cv2.setMouseCallback("Annotate Centers", _mouse_callback)

    def _render(img, fid: int) -> object:
        display = img.copy()

        # Draw bbox overlay (from-track mode)
        if fid in bboxes:
            _draw_bbox_overlay(display, bboxes[fid])

        # Draw hint (scratch mode with tracking hints, when not yet annotated)
        if not seeded and fid in hints and fid not in annotations:
            _draw_center(display, *hints[fid], (255, 200, 0), radius=12, label="TRACK")

        # Draw annotation / auto center
        if fid in annotations:
            cx, cy = annotations[fid]
            is_corrected = fid in corrected_frames
            if is_corrected:
                color = (0, 255, 0)       # green
                label = "CORRECTED"
            elif seeded and interpolated_flags.get(fid, False):
                color = (0, 220, 255)     # yellow
                label = "AUTO (interpolated)"
            elif seeded:
                color = (255, 220, 0)     # cyan
                label = "AUTO"
            else:
                color = (0, 255, 0)       # green
                label = "ANNOTATED"
            _draw_center(display, cx, cy, color)
            # Show hint in background if it differs from annotation (scratch + hints)
            if not seeded and fid in hints:
                hx, hy = hints[fid]
                if abs(hx - cx) > 5 or abs(hy - cy) > 5:
                    _draw_center(display, hx, hy, (255, 200, 0), radius=12, label="TRACK")
        elif fid not in annotations:
            label = "REMOVED" if seeded else "---"
        else:
            label = ""

        bar = (
            f"Frame {fid}  [{idx + 1}/{total_display}]  "
            f"{'Corrected' if seeded else 'Annotated'}: "
            f"{len(corrected_frames) if seeded else len(annotations)}  |  {label}"
        )
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
        bar_color = (0, 255, 0) if fid in corrected_frames else (0, 255, 255)
        cv2.putText(display, bar, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
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

        display = _render(img, frame_id)
        cv2.imshow("Annotate Centers", display)
        _click_pos = None

        while True:
            key = cv2.waitKey(30) & 0xFF

            # Handle click → override / set center
            if _click_pos is not None:
                cx, cy = _click_pos
                annotations[frame_id] = (float(cx), float(cy))
                corrected_frames.add(frame_id)
                _click_pos = None
                display = _render(img, frame_id)
                cv2.imshow("Annotate Centers", display)

            # SPACE or RIGHT → accept and advance
            if key == ord(" ") or key == 83 or key == 3:
                # Auto-accept hint if no annotation yet (scratch + hints mode)
                if frame_id not in annotations and frame_id in hints:
                    annotations[frame_id] = hints[frame_id]
                idx += 1
                break

            # LEFT → go back
            if key == 81 or key == 2:
                idx = max(0, idx - 1)
                break

            # 'u' → undo correction / annotation
            if key == ord("u"):
                corrected_frames.discard(frame_id)
                if seeded:
                    ac = auto_centers.get(frame_id)
                    if ac:
                        annotations[frame_id] = ac
                    else:
                        annotations.pop(frame_id, None)
                else:
                    annotations.pop(frame_id, None)
                display = _render(img, frame_id)
                cv2.imshow("Annotate Centers", display)

            # 'r' → remove frame from annotations
            if key == ord("r"):
                annotations.pop(frame_id, None)
                corrected_frames.discard(frame_id)
                display = _render(img, frame_id)
                cv2.imshow("Annotate Centers", display)

            # 's' → skip (no annotation, advance)
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

            # 'a' → accept all remaining and quit (from-track mode)
            if key == ord("a") and seeded:
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
        print("Usage: uv run python -m src.tracking.annotate_centers <frames_dir> <output_csv> [options]")
        print()
        print("Options:")
        print("  --step=N                Annotate every N-th frame (default: 1)")
        print("  --track-id=N            Track ID written to CSV (default: 1)")
        print("  --from-track=<dir>      Pre-populate from tracking dir (review mode)")
        print("  --tracking=<json>       Show tracking hints (scratch mode)")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    kwargs: dict = {}
    for arg in sys.argv[3:]:
        if arg.startswith("--step="):
            kwargs["frame_step"] = int(arg.split("=")[1])
        elif arg.startswith("--track-id="):
            kwargs["track_id"] = int(arg.split("=")[1])
        elif arg.startswith("--from-track="):
            kwargs["from_track"] = Path(arg.split("=", 1)[1])
        elif arg.startswith("--tracking="):
            tracking_path = Path(arg.split("=", 1)[1])
            if not tracking_path.exists():
                print(f"Warning: tracking file not found: {tracking_path}")
            else:
                centers, _, _ = _load_tracking(tracking_path)
                kwargs["tracking_hints"] = centers

    annotate_video(frames_dir, output_csv, **kwargs)
