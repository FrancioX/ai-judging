"""Analyze which segmentation tracks match the GT skier for each annotated video."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze(stem: str) -> None:
    seg_path = ROOT / "output" / "segmentation" / stem / "segmentation.json"
    gt_path = ROOT / "annotations" / "tracking" / stem / "gt_centers.csv"
    if not seg_path.exists() or not gt_path.exists():
        print(f"SKIP {stem}: missing files")
        return

    with open(seg_path) as f:
        seg = json.load(f)

    gt: dict[int, tuple[float, float]] = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            gt[int(parts[0])] = (float(parts[2]), float(parts[3]))

    # Gather per-track observations
    tracks: dict[int, list[dict]] = {}
    for fr in seg["frames"]:
        fid = fr["frame_id"]
        for p in fr.get("persons", []):
            tid = p.get("track_id")
            if tid is None:
                continue
            bbox = p["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            tracks.setdefault(tid, []).append(
                {"fid": fid, "cx": cx, "cy": cy, "conf": p["confidence"]}
            )

    print(f"\n{'='*80}")
    print(f"VIDEO: {stem}")
    print(f"  GT frames: {len(gt)}  (range {min(gt)}-{max(gt)})")
    print(f"  Tracks: {len(tracks)}")
    print(
        f"  {'Track':>6} | {'N_det':>6} | {'First':>6} | {'Last':>6} "
        f"| {'AvgDistGT':>10} | {'%<50px':>6} | {'GToverlap':>9}"
    )
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}-+-{'-'*9}")

    ranked = []
    for tid in sorted(tracks.keys()):
        obs = tracks[tid]
        n = len(obs)
        first = obs[0]["fid"]
        last = obs[-1]["fid"]

        dists = []
        for o in obs:
            if o["fid"] in gt:
                gcx, gcy = gt[o["fid"]]
                d = math.sqrt((o["cx"] - gcx) ** 2 + (o["cy"] - gcy) ** 2)
                dists.append(d)

        if dists:
            avg_d = sum(dists) / len(dists)
            close_pct = sum(1 for d in dists if d < 50) / len(dists) * 100
            print(
                f"  {tid:6d} | {n:6d} | {first:6d} | {last:6d} "
                f"| {avg_d:10.1f} | {close_pct:5.0f}% | {len(dists):9d}"
            )
            ranked.append((avg_d, tid, n, close_pct, len(dists)))
        else:
            print(
                f"  {tid:6d} | {n:6d} | {first:6d} | {last:6d} "
                f"| {'---':>10} | {'---':>6} | {'0':>9}"
            )

    if ranked:
        ranked.sort()
        print(f"\n  Best track by GT proximity: track {ranked[0][1]} "
              f"(avg {ranked[0][0]:.1f}px, {ranked[0][3]:.0f}% <50px, "
              f"{ranked[0][4]} GT overlaps)")
        # Show top-3
        for i, (avg_d, tid, n, pct, noverlap) in enumerate(ranked[:5]):
            print(
                f"    #{i+1}: track {tid} — avg {avg_d:.1f}px, "
                f"{pct:.0f}% <50px, {noverlap} GT pts, {n} detections"
            )


def main() -> None:
    stems = [
        "VERBIER FREERIDE WEEK QUALIFIER 4__1_Ski Men_Arno Vuarnier_58_Switzerland_91.33",
        "VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89",
        "VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86",
    ]
    for s in stems:
        analyze(s)


if __name__ == "__main__":
    main()
