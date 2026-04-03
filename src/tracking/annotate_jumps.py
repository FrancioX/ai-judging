"""Jump interval annotation tool using OpenCV.

Navigate through frames and mark jump start/end boundaries. Outputs a
``gt_jumps.csv`` with both tracking frame IDs and source video frame numbers.

Controls
--------
- **RIGHT / SPACE**: advance one frame
- **LEFT**: go back one frame
- **j / k**: jump forward / backward 10 frames
- **s**: set jump START at current frame (clears any pending start)
- **e**: set jump END at current frame (completes pending jump)
- **d**: delete the last completed jump
- **u**: cancel pending start (no jump committed)
- **q**: save and quit

The bottom panel shows all completed jumps and the pending start (if any).
Frames that fall inside a completed jump interval are tinted green.

Usage
-----
::

    uv run python -m src.tracking.annotate_jumps \\
        "output/tracking/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_jumps.csv"

    # With velocity predictions as hints
    uv run python -m src.tracking.annotate_jumps \\
        "output/tracking/<video_stem>" \\
        "annotations/tracking/<video_stem>/gt_jumps.csv" \\
        --velocity="output/velocity/<video_stem>"
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tracking(tracking_dir: Path) -> list[dict]:
    """Load frames list from tracking.json.

    Returns a list of dicts with keys: frame_id, frame_file, source_frame_id.
    """
    manifest = tracking_dir / "tracking.json"
    if not manifest.exists():
        raise FileNotFoundError(f"tracking.json not found at {manifest}")
    data = json.loads(manifest.read_text())
    frames = []
    for f in data.get("frames", []):
        fid = f["frame_id"]
        frame_file = f.get("frame_file", "")
        try:
            src_id = int(Path(frame_file).stem.split("_")[-1])
        except (ValueError, IndexError):
            src_id = fid
        frames.append({"frame_id": fid, "frame_file": frame_file, "source_frame_id": src_id})
    return frames


def _load_velocity_hints(velocity_dir: Path) -> list[tuple[int, int]]:
    """Load predicted jump intervals from velocity.json as (start, end) tuples."""
    vpath = velocity_dir / "velocity.json"
    if not vpath.exists():
        return []
    data = json.loads(vpath.read_text())
    return [
        (j["start_frame_id"], j["end_frame_id"])
        for j in data.get("jumps", [])
    ]


def _load_gt(gt_path: Path) -> list[tuple[int, int, int, int]]:
    """Load existing gt_jumps.csv → list of (start_frame, end_frame, start_src, end_src)."""
    jumps = []
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f):
            s, e = int(row["start_frame"]), int(row["end_frame"])
            ss = int(row.get("start_source_frame", s))
            es = int(row.get("end_source_frame", e))
            if e > s:
                jumps.append((s, e, ss, es))
    return jumps


def _save_gt(gt_path: Path, jumps: list[tuple[int, int, int, int]]) -> None:
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["jump_id", "start_frame", "end_frame", "start_source_frame", "end_source_frame"])
        for i, (s, e, ss, es) in enumerate(jumps, 1):
            writer.writerow([i, s, e, ss, es])
    print(f"\nSaved {len(jumps)} jumps → {gt_path}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PANEL_H = 90  # height of the info panel below the frame
_TINT_ALPHA = 0.25
_GREEN = (0, 220, 80)
_CYAN = (220, 200, 0)
_ORANGE = (0, 140, 255)
_RED = (0, 60, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GRAY = (120, 120, 120)


def _render(
    img,
    frame_id: int,
    src_id: int,
    idx: int,
    total: int,
    jumps: list[tuple[int, int, int, int]],
    pending_start: int | None,
    pending_src: int | None,
    hints: list[tuple[int, int]],
) -> object:
    display = img.copy()

    # Tint frame green if inside a completed jump
    in_jump = any(s <= frame_id <= e for s, e, *_ in jumps)
    if in_jump:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), (0, 180, 60), -1)
        cv2.addWeighted(overlay, _TINT_ALPHA, display, 1 - _TINT_ALPHA, 0, display)

    # Tint frame yellow-ish if pending start has been set
    if pending_start is not None and frame_id >= pending_start:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), (0, 200, 255), -1)
        cv2.addWeighted(overlay, _TINT_ALPHA, display, 1 - _TINT_ALPHA, 0, display)

    # Hint intervals: draw thin cyan bar at top
    for hs, he in hints:
        if hs <= frame_id <= he:
            cv2.rectangle(display, (0, 0), (display.shape[1], 6), _CYAN, -1)
            break

    # Top status bar
    jump_num = next((i + 1 for i, (s, e, *_) in enumerate(jumps) if s <= frame_id <= e), None)
    if jump_num:
        status = f"JUMP {jump_num}"
        bar_color = _GREEN
    elif pending_start is not None and frame_id >= pending_start:
        status = f"PENDING  [start={pending_start}]"
        bar_color = _ORANGE
    elif pending_start is not None:
        status = f"PENDING start={pending_start}  (navigate to end, press e)"
        bar_color = _ORANGE
    else:
        status = "---"
        bar_color = _GRAY

    bar_text = f"Frame {frame_id:4d}  (src {src_id:5d})  [{idx + 1}/{total}]  Jumps: {len(jumps)}   {status}"
    cv2.rectangle(display, (0, 0), (display.shape[1], 40), _BLACK, -1)
    cv2.putText(display, bar_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_color, 2)

    # Bottom panel
    h, w = display.shape[:2]
    panel = np.zeros((_PANEL_H, w, 3), dtype=np.uint8)

    # Controls hint
    cv2.putText(panel, "s=start  e=end  d=del last  u=cancel  j/k=±10  q=save+quit",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GRAY, 1)

    # List completed jumps (up to 6 per row)
    jump_strs = [f"J{i+1}:[{s}-{e}]" for i, (s, e, *_) in enumerate(jumps)]
    row_y = 42
    x = 10
    for js in jump_strs:
        tw = cv2.getTextSize(js, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        if x + tw + 14 > w:
            row_y += 22
            x = 10
        cv2.putText(panel, js, (x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GREEN, 1)
        x += tw + 14

    if pending_start is not None:
        ps_str = f"PENDING: [{pending_start} → ?]"
        cv2.putText(panel, ps_str, (10, row_y + 22 if jump_strs else row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _ORANGE, 1)

    if hints:
        hint_strs = " ".join(f"H{i+1}:[{s}-{e}]" for i, (s, e) in enumerate(hints))
        cv2.putText(panel, f"hints: {hint_strs}", (10, _PANEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _CYAN, 1)

    return np.vstack([display, panel])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_jumps(
    tracking_dir: Path,
    output_csv: Path,
    *,
    frames_dir: Path | None = None,
    velocity_dir: Path | None = None,
) -> Path:
    """Open an interactive window to annotate jump intervals.

    Parameters
    ----------
    tracking_dir : Path
        Directory containing ``tracking.json`` (and optionally ``crops/``).
    output_csv : Path
        Where to write the annotation CSV.
    frames_dir : Path | None
        Directory containing extracted frames. Defaults to
        ``output/frames/<stem>`` relative to the project root.
    velocity_dir : Path | None
        Directory containing ``velocity.json`` for hint overlays.

    Returns
    -------
    Path
        Path to the saved annotation CSV.
    """
    tracking_dir = Path(tracking_dir)
    output_csv = Path(output_csv)

    # Resolve frames directory
    if frames_dir is None:
        stem = tracking_dir.name
        root = Path(__file__).resolve().parent.parent.parent
        frames_dir = root / "output" / "frames" / stem
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Load frame map from tracking.json
    frame_map = _load_tracking(tracking_dir)  # list of {frame_id, frame_file, source_frame_id}
    frame_by_id: dict[int, dict] = {f["frame_id"]: f for f in frame_map}

    # Resolve frame file paths
    frame_paths: list[tuple[int, int, Path]] = []  # (frame_id, src_id, path)
    for fm in frame_map:
        p = frames_dir / fm["frame_file"]
        if p.exists():
            frame_paths.append((fm["frame_id"], fm["source_frame_id"], p))
    if not frame_paths:
        raise FileNotFoundError(f"No frame files found in {frames_dir}")
    frame_paths.sort(key=lambda x: x[0])

    total = len(frame_paths)

    # Load velocity hints
    hints: list[tuple[int, int]] = []
    if velocity_dir is not None:
        hints = _load_velocity_hints(Path(velocity_dir))
        if hints:
            print(f"Loaded {len(hints)} velocity hint intervals")

    # Resume from existing CSV
    jumps: list[tuple[int, int, int, int]] = []  # (start, end, start_src, end_src)
    if output_csv.exists():
        jumps = _load_gt(output_csv)
        print(f"Resumed {len(jumps)} jumps from {output_csv}")

    pending_start: int | None = None
    pending_src: int | None = None

    print(f"\nJump annotation — {tracking_dir.name}")
    print(f"  Frames : {frames_dir}  ({total} frames)")
    print(f"  Output : {output_csv}")
    print("  Controls: s=start  e=end  d=del-last  u=cancel-pending  j/k=±10  q=save+quit\n")

    cv2.namedWindow("Annotate Jumps", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate Jumps", 1280, 800)

    idx = 0
    while 0 <= idx < total:
        frame_id, src_id, frame_path = frame_paths[idx]
        img = cv2.imread(str(frame_path))
        if img is None:
            idx += 1
            continue

        display = _render(img, frame_id, src_id, idx, total, jumps, pending_start, pending_src, hints)
        cv2.imshow("Annotate Jumps", display)

        key = cv2.waitKey(0) & 0xFF

        # RIGHT or SPACE → advance
        if key in (83, ord(" "), 3):
            idx += 1

        # LEFT → back
        elif key in (81, 2):
            idx = max(0, idx - 1)

        # j → +10 frames
        elif key == ord("j"):
            idx = min(total - 1, idx + 10)

        # k → -10 frames
        elif key == ord("k"):
            idx = max(0, idx - 10)

        # s → set pending start
        elif key == ord("s"):
            pending_start = frame_id
            pending_src = src_id
            print(f"  Jump start set: frame {frame_id} (src {src_id})")

        # e → complete jump
        elif key == ord("e"):
            if pending_start is None:
                print("  No pending start — press s first")
            elif frame_id <= pending_start:
                print(f"  End frame ({frame_id}) must be after start ({pending_start})")
            else:
                jumps.append((pending_start, frame_id, pending_src, src_id))
                print(f"  Jump {len(jumps)} added: [{pending_start}–{frame_id}] (src [{pending_src}–{src_id}])")
                pending_start = None
                pending_src = None

        # d → delete last jump
        elif key == ord("d"):
            if jumps:
                removed = jumps.pop()
                print(f"  Deleted last jump: [{removed[0]}–{removed[1]}]")
            else:
                print("  No jumps to delete")

        # u → cancel pending start
        elif key == ord("u"):
            if pending_start is not None:
                print(f"  Cancelled pending start (was frame {pending_start})")
                pending_start = None
                pending_src = None

        # q → save and quit
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    _save_gt(output_csv, jumps)
    return output_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]

    if len(argv) < 2:
        print(__doc__)
        print("Usage: uv run python -m src.tracking.annotate_jumps <tracking_dir> <output_csv> [options]")
        print()
        print("Options:")
        print("  --frames=<dir>      Override frames directory")
        print("  --velocity=<dir>    Load velocity.json hints")
        sys.exit(1)

    tracking_dir = Path(argv[0])
    output_csv = Path(argv[1])

    kwargs: dict = {}
    for arg in argv[2:]:
        if arg.startswith("--frames="):
            kwargs["frames_dir"] = Path(arg.split("=", 1)[1])
        elif arg.startswith("--velocity="):
            kwargs["velocity_dir"] = Path(arg.split("=", 1)[1])

    annotate_jumps(tracking_dir, output_csv, **kwargs)


if __name__ == "__main__":
    main()
