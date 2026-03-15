"""Venue annotation tool — mark athlete position on the venue image.

Shows a side-by-side window:
  LEFT panel  : current video frame (reference — see where the athlete is)
  RIGHT panel : venue image (click here to mark the athlete's position)

Annotations are saved as a CSV with columns: frame_id, venue_x, venue_y

Usage
-----
::

    # Annotate from scratch, every 10th frame
    uv run python scripts/annotate_venue.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4" \\
        --step 10

    # Resume an existing session
    uv run python scripts/annotate_venue.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4" \\
        --step 10

    # Use a specific output path
    uv run python scripts/annotate_venue.py \\
        "raw_videos/Ski Men_2_89_Andreas Bakke.mp4" \\
        --output "annotations/venue/my_run/gt_venue.csv"

Controls
--------
  Left-click on right panel  : set / override athlete position on venue
  SPACE / RIGHT arrow        : accept and advance to next frame
  LEFT arrow                 : go back one frame
  j / k                      : jump forward / backward 10 frames
  u                          : undo annotation on current frame
  q                          : save and quit
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Module-level mouse state
# ---------------------------------------------------------------------------
_click_pos: tuple[int, int] | None = None
_left_panel_w: int = 0  # set before each window interaction


def _mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    """Record left-click only if it falls in the right (venue) panel."""
    global _click_pos
    if event == cv2.EVENT_LBUTTONDOWN and x >= _left_panel_w:
        _click_pos = (x, y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing(csv_path: Path) -> dict[int, tuple[float, float]]:
    """Load any existing annotations from CSV."""
    annotations: dict[int, tuple[float, float]] = {}
    if not csv_path.exists():
        return annotations
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                annotations[int(row[0])] = (float(row[1]), float(row[2]))
            except (ValueError, IndexError):
                continue
    return annotations


def _save(csv_path: Path, annotations: dict[int, tuple[float, float]]) -> None:
    """Write annotations to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        f.write("# frame_id,venue_x,venue_y\n")
        writer = csv.writer(f)
        for fid in sorted(annotations):
            vx, vy = annotations[fid]
            writer.writerow([fid, f"{vx:.1f}", f"{vy:.1f}"])
    print(f"\nSaved {len(annotations)} annotations → {csv_path}")


def _draw_marker(
    img: np.ndarray,
    cx: int,
    cy: int,
    color: tuple[int, int, int],
    radius: int = 8,
) -> None:
    cv2.circle(img, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius + 3, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def _draw_trail(
    img: np.ndarray,
    trail: list[tuple[int, int]],
    scale_x: float,
    scale_y: float,
) -> None:
    """Draw fading trail of past annotations in display coordinates."""
    if len(trail) < 2:
        return
    n = len(trail)
    for i in range(len(trail) - 1):
        alpha = float(i + 1) / float(max(1, n - 1))
        c = int(40 + 200 * alpha)
        color = (c, c, 255)
        p0 = (int(trail[i][0] * scale_x), int(trail[i][1] * scale_y))
        p1 = (int(trail[i + 1][0] * scale_x), int(trail[i + 1][1] * scale_y))
        cv2.line(img, p0, p1, color, thickness=1 if alpha < 0.5 else 2, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main annotate function
# ---------------------------------------------------------------------------

def annotate_venue(
    video_path: Path,
    venue_image_path: Path,
    output_csv: Path,
    frame_root: Path,
    *,
    step: int = 10,
    display_height: int = 720,
) -> Path:
    """Run the interactive venue annotation session.

    Parameters
    ----------
    video_path : Path
        Raw video path (used to derive the video stem for locating frames).
    venue_image_path : Path
        Venue reference image to click on.
    output_csv : Path
        Output CSV path.
    frame_root : Path
        Root directory containing extracted frames per video stem.
    step : int
        Annotate every N-th frame.
    display_height : int
        Target display height in pixels for both panels.

    Returns
    -------
    Path
        Path to the saved annotation CSV.
    """
    global _click_pos, _left_panel_w

    video_stem = video_path.stem
    frame_dir = frame_root / video_stem

    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    if not venue_image_path.exists():
        raise FileNotFoundError(f"Venue image not found: {venue_image_path}")

    # Load frames
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frame_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    def _frame_id(p: Path) -> int:
        return int(p.stem.split("_")[-1])

    display_indices = list(range(0, len(frame_files), max(1, step)))
    total = len(display_indices)

    # Load venue image (kept at full res for accurate coordinate mapping)
    venue_full = cv2.imread(str(venue_image_path), cv2.IMREAD_COLOR)
    if venue_full is None:
        raise RuntimeError(f"Cannot read venue image: {venue_image_path}")
    venue_h_full, venue_w_full = venue_full.shape[:2]

    # Compute display scale factors
    # Left panel: scale a sample frame to display_height
    sample = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError("Cannot read sample frame")
    frame_h_full, frame_w_full = sample.shape[:2]
    left_scale = display_height / frame_h_full
    left_w = int(round(frame_w_full * left_scale))

    # Right panel: venue scaled to display_height
    right_scale_y = display_height / venue_h_full
    right_scale_x = right_scale_y  # uniform scale
    right_w = int(round(venue_w_full * right_scale_x))

    venue_display = cv2.resize(venue_full, (right_w, display_height), interpolation=cv2.INTER_AREA)

    # coordinate helpers: display → full
    def _display_to_venue(dx: int, dy: int) -> tuple[float, float]:
        """Convert right-panel display coords to full-resolution venue coords."""
        panel_x = dx - left_w  # offset for left panel
        vx = float(np.clip(panel_x / right_scale_x, 0, venue_w_full - 1))
        vy = float(np.clip(dy / right_scale_y, 0, venue_h_full - 1))
        return vx, vy

    def _venue_to_display(vx: float, vy: float) -> tuple[int, int]:
        """Convert full-res venue coords to right-panel display coords."""
        dx = int(round(vx * right_scale_x))
        dy = int(round(vy * right_scale_y))
        return dx, dy

    _left_panel_w = left_w

    # Resume existing annotations
    annotations: dict[int, tuple[float, float]] = _load_existing(output_csv)
    if annotations:
        print(f"Resumed {len(annotations)} existing annotations from {output_csv}")

    print("\nVenue annotation tool")
    print(f"  Video stem : {video_stem}")
    print(f"  Frames     : {len(frame_files)}  |  step={step}  →  {total} to annotate")
    print(f"  Venue      : {venue_w_full}×{venue_h_full}")
    print(f"  Output     : {output_csv}")
    print("\nControls: click venue=mark  SPACE/RIGHT=next  LEFT=prev  j/k=+/-10  u=undo  q=save+quit\n")

    cv2.namedWindow("Annotate Venue", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate Venue", left_w + right_w, display_height)
    cv2.setMouseCallback("Annotate Venue", _mouse_callback)

    def _render(frame_bgr: np.ndarray, frame_id: int, idx: int) -> np.ndarray:
        """Compose side-by-side display frame."""
        # Left: scaled video frame
        left = cv2.resize(frame_bgr, (left_w, display_height), interpolation=cv2.INTER_AREA)

        # Right: venue with trail + current marker
        right = venue_display.copy()

        # Draw trail (last 50 annotated positions before current)
        sorted_fids = sorted(f for f in annotations if f < frame_id)
        trail_full = [annotations[f] for f in sorted_fids[-50:]]
        if trail_full:
            _draw_trail(right, [_venue_to_display(vx, vy) for vx, vy in trail_full],
                        scale_x=1.0, scale_y=1.0)

        # Draw current annotation
        if frame_id in annotations:
            vx, vy = annotations[frame_id]
            dx, dy = _venue_to_display(vx, vy)
            _draw_marker(right, dx, dy, (0, 80, 255))  # red dot

        # Compose
        canvas = np.hstack((left, right))

        # HUD
        annotated_count = len(annotations)
        label = "ANNOTATED" if frame_id in annotations else "---"
        bar = (
            f"Frame {frame_id}  [{idx + 1}/{total}]  "
            f"Annotated: {annotated_count}  |  {label}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (0, 0, 0), -1)
        bar_color = (0, 255, 100) if frame_id in annotations else (100, 100, 100)
        cv2.putText(canvas, bar, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

        # Vertical divider
        cv2.line(canvas, (left_w, 0), (left_w, display_height), (80, 80, 80), 1)

        # Label panels
        cv2.putText(canvas, "VIDEO FRAME (reference)", (10, display_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.putText(canvas, "VENUE MAP (click to place)", (left_w + 10, display_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        return canvas

    idx = 0
    while 0 <= idx < total:
        fi = display_indices[idx]
        frame_path = frame_files[fi]
        frame_id = _frame_id(frame_path)

        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            idx += 1
            continue

        canvas = _render(frame_bgr, frame_id, idx)
        cv2.imshow("Annotate Venue", canvas)
        _click_pos = None

        while True:
            key = cv2.waitKey(30) & 0xFF

            # Mouse click → set venue position
            if _click_pos is not None:
                vx, vy = _display_to_venue(_click_pos[0], _click_pos[1])
                annotations[frame_id] = (vx, vy)
                _click_pos = None
                canvas = _render(frame_bgr, frame_id, idx)
                cv2.imshow("Annotate Venue", canvas)

            # SPACE or RIGHT → advance
            if key in (ord(" "), 83, 3):
                idx += 1
                break

            # LEFT → previous
            if key in (81, 2):
                idx = max(0, idx - 1)
                break

            # j → +10
            if key == ord("j"):
                idx = min(total - 1, idx + 10)
                break

            # k → -10
            if key == ord("k"):
                idx = max(0, idx - 10)
                break

            # u → undo
            if key == ord("u"):
                annotations.pop(frame_id, None)
                canvas = _render(frame_bgr, frame_id, idx)
                cv2.imshow("Annotate Venue", canvas)

            # q → save and quit
            if key == ord("q"):
                idx = total
                break

    cv2.destroyAllWindows()
    _save(output_csv, annotations)
    return output_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Annotate athlete positions on a venue image by clicking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", type=Path, help="Path to raw video (used for stem → frame dir).")
    p.add_argument("--venue-image", type=Path, default=Path("venue_image.png"),
                   help="Venue reference image to click on (default: venue_image.png).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path. Default: annotations/venue/<stem>/gt_venue.csv")
    p.add_argument("--frame-root", type=Path, default=Path("output/frames"),
                   help="Root dir containing extracted frames (default: output/frames).")
    p.add_argument("--step", type=int, default=10,
                   help="Annotate every N-th frame (default: 10).")
    p.add_argument("--display-height", type=int, default=720,
                   help="Display window height in pixels (default: 720).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    video_path = args.video
    video_stem = video_path.stem

    output_csv = args.output or Path(f"annotations/venue/{video_stem}/gt_venue.csv")

    annotate_venue(
        video_path=video_path,
        venue_image_path=args.venue_image,
        output_csv=output_csv,
        frame_root=args.frame_root,
        step=args.step,
        display_height=args.display_height,
    )


if __name__ == "__main__":
    main()
