"""Diagnose remaining identity switches in tracking results."""
from __future__ import annotations

import csv
import json
import math
import sys

VIDEOS = [
    "VERBIER FREERIDE WEEK QUALIFIER 4__1_Ski Men_Arno Vuarnier_58_Switzerland_91.33",
    "VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89",
    "VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86",
]

for stem in VIDEOS:
    print(f"\n{'='*70}")
    print(f"  {stem}")
    print(f"{'='*70}")

    with open(f"output/tracking/{stem}/tracking.json") as f:
        tracking = json.load(f)

    gt: dict[int, tuple[float, float]] = {}
    with open(f"annotations/tracking/{stem}/gt_centers.csv") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            fid_gt = int(parts[0])
            gt[fid_gt] = (float(parts[2]), float(parts[3]))

    big_errors: list[tuple] = []
    for fr in tracking["frames"]:
        fid = fr["frame_id"]
        if fid not in gt:
            continue
        bbox = fr["bbox"]
        pred_cx = (bbox[0] + bbox[2]) / 2
        pred_cy = (bbox[1] + bbox[3]) / 2
        gt_cx, gt_cy = gt[fid]
        err = math.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)
        if err > 100:
            big_errors.append((
                fid, err, fr["detected"], fr.get("track_id", -1),
                pred_cx, pred_cy, gt_cx, gt_cy,
            ))

    print(f"Frames with >100px error: {len(big_errors)} / {len(gt)} GT keyframes")

    if not big_errors:
        print("  No large errors!")
        continue

    # Group into contiguous runs (gap < 20 frames)
    runs: list[list[tuple]] = []
    current_run = [big_errors[0]]
    for i in range(1, len(big_errors)):
        if big_errors[i][0] - big_errors[i - 1][0] < 20:
            current_run.append(big_errors[i])
        else:
            runs.append(current_run)
            current_run = [big_errors[i]]
    runs.append(current_run)

    print(f"Identity switch episodes: {len(runs)}")
    for i, run in enumerate(runs):
        start_fid = run[0][0]
        end_fid = run[-1][0]
        mean_err = sum(e[1] for e in run) / len(run)
        n_det = sum(1 for e in run if e[2])
        print(f"\n  Episode {i + 1}: frames {start_fid}-{end_fid} "
              f"({end_fid - start_fid + 1} frames), "
              f"mean_err={mean_err:.0f}px, "
              f"{n_det}/{len(run)} detected")
        for e in run[:5]:
            print(f"    fid={e[0]}: err={e[1]:.0f}px, det={e[2]}, "
                  f"tid={e[3]}, pred=({e[4]:.0f},{e[5]:.0f}), "
                  f"gt=({e[6]:.0f},{e[7]:.0f})")
        if len(run) > 5:
            print(f"    ... ({len(run) - 5} more)")

    # Also check: how many conflict frames had >1 candidate in segmentation?
    with open(f"output/segmentation/{stem}/segmentation.json") as f:
        seg = json.load(f)

    error_fids = {e[0] for e in big_errors}
    multi_person_in_error = 0
    for sf in seg["frames"]:
        if sf["frame_id"] in error_fids:
            if len(sf.get("persons", [])) > 1:
                multi_person_in_error += 1

    print(f"\n  Error frames with multiple persons in segmentation: "
          f"{multi_person_in_error}/{len(big_errors)}")
