"""Interactive tool for marking the approximate start area on the venue image.

The user drags a rectangle around the region of the venue image where athletes
begin their run.  This region is saved once per venue and is automatically
used by ``venue_registration.py`` as the ``search_region`` constraint —
eliminating false-positive template matches in other parts of the venue.

Usage
-----
::

    uv run python scripts/annotate_venue_start.py
    uv run python scripts/annotate_venue_start.py --venue-image venue_image.png

Controls
--------
- **Drag** left mouse button → draw / redraw rectangle
- **Enter** or **Space** → confirm and save
- **r** → reset (clear current rectangle)
- **q** / **Esc** → quit without saving

Output
------
``annotations/venue/start_region.json`` — venue pixel coordinates of the
selected rectangle, plus the venue image dimensions for sanity-checking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAVE_PATH = Path("annotations/venue/start_region.json")
MAX_DISPLAY_W = 1400
MAX_DISPLAY_H = 900
WINDOW = "Venue start area — drag rectangle, Enter=save, r=reset, q=quit"

COLOR_RECT   = (0, 200, 50)    # green while drawing
COLOR_DONE   = (0, 120, 255)   # orange when confirmed
COLOR_TEXT   = (255, 255, 255)
COLOR_BG     = (30,  30,  30)


# ---------------------------------------------------------------------------
# Annotation state
# ---------------------------------------------------------------------------

class _State:
    def __init__(self) -> None:
        self.pt1: tuple[int, int] | None = None   # drag start (display coords)
        self.pt2: tuple[int, int] | None = None   # drag end   (display coords)
        self.dragging: bool = False
        self.confirmed: bool = False


def _mouse_cb(event: int, x: int, y: int, flags: int, state: _State) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.pt1 = (x, y)
        state.pt2 = (x, y)
        state.dragging = True
        state.confirmed = False
    elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
        state.pt2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state.pt2 = (x, y)
        state.dragging = False


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _display_to_venue(
    pt: tuple[int, int],
    scale: float,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int]:
    """Convert a display-space point to full venue image coordinates."""
    vx = int(round((pt[0] - offset_x) / scale))
    vy = int(round((pt[1] - offset_y) / scale))
    return vx, vy


def _normalise_rect(
    pt1: tuple[int, int],
    pt2: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) with x1<x2, y1<y2."""
    x1, x2 = sorted([pt1[0], pt2[0]])
    y1, y2 = sorted([pt1[1], pt2[1]])
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render(
    base: np.ndarray,
    state: _State,
    offset_x: int,
    offset_y: int,
    scale: float,
    venue_w: int,
    venue_h: int,
    display_w: int,
    display_h: int,
    existing: dict | None,
) -> np.ndarray:
    canvas = base.copy()

    # Draw existing saved region (grey) if no new rect drawn yet
    if existing and state.pt1 is None:
        ex1 = int(round(existing["x1"] * scale)) + offset_x
        ey1 = int(round(existing["y1"] * scale)) + offset_y
        ex2 = int(round(existing["x2"] * scale)) + offset_x
        ey2 = int(round(existing["y2"] * scale)) + offset_y
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (120, 120, 120), 2)
        cv2.putText(canvas, "existing", (ex1, max(12, ey1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

    # Draw current rectangle
    if state.pt1 and state.pt2:
        dx1, dy1, dx2, dy2 = _normalise_rect(state.pt1, state.pt2)
        color = COLOR_DONE if state.confirmed else COLOR_RECT
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), color, 2)

        # Convert to venue coords for readout
        vx1, vy1 = _display_to_venue((dx1, dy1), scale, offset_x, offset_y)
        vx2, vy2 = _display_to_venue((dx2, dy2), scale, offset_x, offset_y)
        vx1 = max(0, min(vx1, venue_w))
        vy1 = max(0, min(vy1, venue_h))
        vx2 = max(0, min(vx2, venue_w))
        vy2 = max(0, min(vy2, venue_h))

        label = f"({vx1},{vy1}) – ({vx2},{vy2})  [{vx2-vx1}×{vy2-vy1}]"
        if state.confirmed:
            label = "SAVED  " + label
        lx = max(dx1, 4)
        ly = max(dy1 - 8, 14)
        cv2.putText(canvas, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Status bar
    bar_y = display_h - 1
    cv2.rectangle(canvas, (0, bar_y - 22), (display_w, bar_y), COLOR_BG, -1)
    hint = "Drag=draw rect  |  Enter=save  |  r=reset  |  q=quit"
    cv2.putText(canvas, hint, (8, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def annotate_venue_start(
    venue_image_path: str | Path = "venue_image.png",
    save_path: str | Path = SAVE_PATH,
) -> dict | None:
    """Open the venue image, let the user drag a start-area rectangle, save it.

    Returns the saved region dict, or None if the user quit without saving.
    """
    venue_path = Path(venue_image_path)
    if not venue_path.exists():
        raise FileNotFoundError(f"Venue image not found: {venue_path}")

    venue_bgr = cv2.imread(str(venue_path), cv2.IMREAD_COLOR)
    if venue_bgr is None:
        raise RuntimeError(f"Cannot read venue image: {venue_path}")
    venue_h, venue_w = venue_bgr.shape[:2]

    # Scale to fit display while preserving aspect ratio
    scale_w = MAX_DISPLAY_W / venue_w
    scale_h = MAX_DISPLAY_H / venue_h
    scale = min(scale_w, scale_h, 1.0)   # never upscale
    disp_w = int(round(venue_w * scale))
    disp_h = int(round(venue_h * scale)) + 24  # +24 for status bar

    venue_small = cv2.resize(venue_bgr, (disp_w, disp_h - 24), interpolation=cv2.INTER_AREA)

    # Centre the venue image in a fixed-size canvas
    canvas_base = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    canvas_base[:disp_h - 24, :disp_w] = venue_small
    offset_x, offset_y = 0, 0

    # Load existing saved region if any
    save_path = Path(save_path)
    existing: dict | None = None
    if save_path.exists():
        try:
            existing = json.loads(save_path.read_text())
            print(f"[annotate] Existing region: {existing}")
        except Exception:
            pass

    state = _State()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, disp_w, disp_h)
    cv2.setMouseCallback(WINDOW, _mouse_cb, state)

    print(f"[annotate] Venue image: {venue_path}  ({venue_w}×{venue_h}px)")
    print(f"[annotate] Display scale: {scale:.3f}x  ({disp_w}×{disp_h - 24}px)")
    print("[annotate] Drag a rectangle around the athlete start area,")
    print("           then press Enter to save.  'r' to reset, 'q' to quit.")

    saved_region: dict | None = None

    while True:
        frame = _render(
            canvas_base, state,
            offset_x, offset_y, scale,
            venue_w, venue_h,
            disp_w, disp_h,
            existing,
        )
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space → save
            if state.pt1 and state.pt2:
                dx1, dy1, dx2, dy2 = _normalise_rect(state.pt1, state.pt2)
                vx1, vy1 = _display_to_venue((dx1, dy1), scale, offset_x, offset_y)
                vx2, vy2 = _display_to_venue((dx2, dy2), scale, offset_x, offset_y)
                vx1 = max(0, min(vx1, venue_w))
                vy1 = max(0, min(vy1, venue_h))
                vx2 = max(0, min(vx2, venue_w))
                vy2 = max(0, min(vy2, venue_h))
                if vx2 > vx1 and vy2 > vy1:
                    saved_region = {
                        "x1": vx1, "y1": vy1, "x2": vx2, "y2": vy2,
                        "venue_image": str(venue_path),
                        "venue_w": venue_w,
                        "venue_h": venue_h,
                    }
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_text(json.dumps(saved_region, indent=2))
                    state.confirmed = True
                    print(f"[annotate] Saved: {saved_region}")
                    print(f"[annotate] → {save_path}")
                    # Show confirmation for a moment then exit
                    frame = _render(
                        canvas_base, state,
                        offset_x, offset_y, scale,
                        venue_w, venue_h,
                        disp_w, disp_h,
                        existing,
                    )
                    cv2.imshow(WINDOW, frame)
                    cv2.waitKey(800)
                    break
                else:
                    print("[annotate] Rectangle too small — drag again.")
            else:
                print("[annotate] No rectangle drawn yet.")

        elif key == ord("r"):
            state.pt1 = state.pt2 = None
            state.dragging = False
            state.confirmed = False
            print("[annotate] Reset.")

        elif key in (ord("q"), 27):  # q or Esc
            print("[annotate] Quit without saving.")
            break

        # Exit if window was closed
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return saved_region


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mark approximate athlete start area on the venue image."
    )
    p.add_argument(
        "--venue-image", type=Path, default=Path("venue_image.png"),
        help="Path to venue image (default: venue_image.png)",
    )
    p.add_argument(
        "--save-path", type=Path, default=SAVE_PATH,
        help=f"Where to write the region JSON (default: {SAVE_PATH})",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = annotate_venue_start(args.venue_image, args.save_path)
    if result is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
