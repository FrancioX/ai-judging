"""Annotate all tracked videos that don't yet have ground-truth annotations.

Iterates through output/tracking/ in order, skipping stems that already have
annotations/tracking/<stem>/gt_centers.csv, and opens annotate_centers in from-track
mode for each.

Usage
-----
    uv run python annotate_queue.py [--step=N] [--track-id=N]

Options
-------
    --step=N      Review every N-th frame (default: 10)
    --track-id=N  Track ID written to CSV (default: 1)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
TRACKING_OUT = ROOT / "output" / "tracking"
ANNOTATIONS = ROOT / "annotations" / "tracking"

# Parse CLI kwargs to pass through
kwargs: dict = {"frame_step": 10, "track_id": 1}
for arg in sys.argv[1:]:
    if arg.startswith("--step="):
        kwargs["frame_step"] = int(arg.split("=")[1])
    elif arg.startswith("--track-id="):
        kwargs["track_id"] = int(arg.split("=")[1])

from src.tracking.annotate_centers import annotate_video  # noqa: E402

stems = sorted(p.name for p in TRACKING_OUT.iterdir() if p.is_dir())
todo = [s for s in stems if not (ANNOTATIONS / s / "gt_centers.csv").exists()]

if not todo:
    print("All tracked videos already have annotations.")
    sys.exit(0)

print(f"{len(todo)} videos to annotate (out of {len(stems)} total):\n")
for i, s in enumerate(todo, 1):
    print(f"  {i:2d}. {s}")
print()

for i, stem in enumerate(todo, 1):
    tracking_dir = TRACKING_OUT / stem
    output_csv = ANNOTATIONS / stem / "gt_centers.csv"

    print(f"\n{'='*70}")
    print(f"[{i}/{len(todo)}] {stem}")
    print(f"{'='*70}")

    try:
        frames_dir = tracking_dir.parent.parent / "frames" / tracking_dir.name
        annotate_video(frames_dir, output_csv, from_track=tracking_dir, **kwargs)
    except FileNotFoundError as e:
        print(f"  SKIP — {e}")
        continue

print("\nAll done!")
