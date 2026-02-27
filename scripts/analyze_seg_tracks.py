"""Analyze segmentation track fragmentation for all annotated videos."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEMS = [
    "VERBIER FREERIDE WEEK QUALIFIER 4__1_Ski Men_Arno Vuarnier_58_Switzerland_91.33",
    "VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89",
    "VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86",
]


def analyze(stem: str) -> None:
    seg_path = ROOT / "output" / "segmentation" / stem / "segmentation.json"
    if not seg_path.exists():
        print(f"SKIP {stem}")
        return

    with open(seg_path) as f:
        seg = json.load(f)

    print(f"\n{'='*80}")
    print(f"VIDEO: {stem}")
    print(f"Confidence threshold: {seg.get('confidence_threshold')}")
    print(f"Total frames: {seg['n_frames']}")

    n_det = sum(1 for fr in seg["frames"] if fr["detected"])
    print(f"Detected: {n_det}/{seg['n_frames']}")

    # Track coverage
    tracks: dict[int, list[int]] = {}
    for fr in seg["frames"]:
        for p in fr.get("persons", []):
            tid = p.get("track_id")
            if tid is not None:
                tracks.setdefault(tid, []).append(fr["frame_id"])

    print(f"Total tracks: {len(tracks)}")
    for tid in sorted(tracks.keys()):
        fids = tracks[tid]
        gaps = []
        for i in range(1, len(fids)):
            g = fids[i] - fids[i - 1]
            if g > 1:
                gaps.append(g)
        max_gap = max(gaps) if gaps else 0
        span = fids[-1] - fids[0] + 1
        print(
            f"  Track {tid:3d}: {len(fids):4d} dets, "
            f"frames {fids[0]:4d}-{fids[-1]:4d}, "
            f"span={span:4d}, max_gap={max_gap:3d}"
        )

    no_det = sum(1 for fr in seg["frames"] if len(fr.get("persons", [])) == 0)
    one_det = sum(1 for fr in seg["frames"] if len(fr.get("persons", [])) == 1)
    multi_det = sum(1 for fr in seg["frames"] if len(fr.get("persons", [])) >= 2)
    print(f"\nPerson count distribution:")
    print(f"  0 persons: {no_det} frames ({100*no_det/seg['n_frames']:.1f}%)")
    print(f"  1 person:  {one_det} frames ({100*one_det/seg['n_frames']:.1f}%)")
    print(f"  2+ persons: {multi_det} frames ({100*multi_det/seg['n_frames']:.1f}%)")

    # Confidence distribution of all detections
    confs = []
    for fr in seg["frames"]:
        for p in fr.get("persons", []):
            confs.append(p["confidence"])
    if confs:
        confs.sort()
        print(f"\nDetection confidence distribution (n={len(confs)}):")
        print(f"  min={confs[0]:.3f}  p10={confs[len(confs)//10]:.3f}  "
              f"median={confs[len(confs)//2]:.3f}  "
              f"p90={confs[9*len(confs)//10]:.3f}  max={confs[-1]:.3f}")
        # How many would be added at lower thresholds
        for thresh in [0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
            below = sum(1 for c in confs if c < 0.5 and c >= thresh)
            print(f"  Extra detections at conf>={thresh}: {below}")


for stem in STEMS:
    analyze(stem)
