"""Diagnose Arno's regression at merge_score_threshold=0.6 vs 0.5.

Runs tracking at both thresholds, compares candidate pool diagnostics,
identifies swing tracks (included at 0.5, excluded at 0.6), and
checks their proximity to GT centers.

Usage:
    uv run python scripts/diagnose_arno_threshold.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ARNO_STEM = "Ski Men_1_91.33_Arno Vuarnier"
GT_PATH = Path("annotations/tracking") / ARNO_STEM / "gt_centers.csv"
SEG_MANIFEST = Path("output/segmentation") / ARNO_STEM / "segmentation.json"
FRAME_DIR = Path("output/frames") / ARNO_STEM
TRACKING_DIR_BASE = Path("output/tracking") / ARNO_STEM


def _run_tracking(threshold: float) -> dict:
    """Run tracking for Arno at the given threshold; return candidate_pool_stats."""
    from src.pipeline import load_config
    from src.tracking.tracker import track_skier

    config = load_config(Path("config.yaml"))
    trk_cfg = config.get("tracking", {})

    manifest_path = track_skier(
        seg_manifest_path=SEG_MANIFEST,
        frame_dir=FRAME_DIR,
        output_dir=TRACKING_DIR_BASE,
        w_conf=trk_cfg.get("w_conf", 0.3),
        w_center=trk_cfg.get("w_center", 0.5),
        w_length=trk_cfg.get("w_length", 0.2),
        min_track_frames=trk_cfg.get("min_track_frames", 10),
        smooth_window=trk_cfg.get("smooth_window", 5),
        padding_ratio=trk_cfg.get("padding_ratio", 0.15),
        merge_tracks=True,
        merge_score_threshold=threshold,
        merge_min_detection_conf=0.0,
        w_continuity=0.6,
        w_track_stickiness=0.4,
        optical_flow_method="none",
        flow_max_extrapolate_frames=trk_cfg.get("flow_max_extrapolate_frames", 30),
        flow_min_keypoints=trk_cfg.get("flow_min_keypoints", 5),
        identity_guard_enabled=False,
        identity_guard_max_jump_px=100.0,
        identity_guard_reanchor_interval=trk_cfg.get("identity_guard_reanchor_interval", 50),
        identity_guard_reanchor_min_conf=trk_cfg.get("identity_guard_reanchor_min_conf", 0.5),
        identity_guard_max_drift_px=trk_cfg.get("identity_guard_max_drift_px", 200.0),
        w_velocity=trk_cfg.get("w_velocity", 0.4),
        vel_history_len=trk_cfg.get("vel_history_len", 5),
        of_synthetic_confidence=trk_cfg.get("of_synthetic_confidence", 0.3),
        cmc_enabled=False,
        cmc_method="none",
        cmc_exclude_margin=trk_cfg.get("cmc_exclude_margin", 1.5),
        cmc_min_features=trk_cfg.get("cmc_min_features", 20),
        cmc_ransac_threshold=trk_cfg.get("cmc_ransac_threshold", 3.0),
    )
    with open(manifest_path) as f:
        return json.load(f)


def _load_gt() -> dict[int, tuple[float, float]]:
    """Load GT centers; return {frame_id: (cx, cy)} linearly interpolated."""
    sparse: dict[int, tuple[float, float]] = {}
    with open(GT_PATH) as f:
        reader = csv.DictReader(
            (line for line in f if not line.startswith("#")),
            fieldnames=["frame_id", "track_id", "center_x", "center_y"],
        )
        for row in reader:
            sparse[int(row["frame_id"])] = (float(row["center_x"]), float(row["center_y"]))
    if not sparse:
        return {}
    fids = sorted(sparse)
    gt: dict[int, tuple[float, float]] = {}
    # Leading fill
    for fid in range(0, fids[0]):
        gt[fid] = sparse[fids[0]]
    # Interpolate between annotated frames
    for i in range(len(fids) - 1):
        a, b = fids[i], fids[i + 1]
        gt[a] = sparse[a]
        for fid in range(a + 1, b):
            t = (fid - a) / (b - a)
            gt[fid] = (
                sparse[a][0] + t * (sparse[b][0] - sparse[a][0]),
                sparse[a][1] + t * (sparse[b][1] - sparse[a][1]),
            )
    gt[fids[-1]] = sparse[fids[-1]]
    return gt


def _seg_detections_by_track(stem: str) -> dict[int, list[dict]]:
    """Load segmentation detections grouped by ByteTrack track_id."""
    seg_manifest_path = Path("output/segmentation") / stem / "segmentation.json"
    with open(seg_manifest_path) as f:
        seg = json.load(f)
    by_track: dict[int, list[dict]] = {}
    for frame in seg.get("frames", []):
        fid = frame["frame_id"]
        for det in frame.get("persons", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            by_track.setdefault(tid, []).append({
                "frame_id": fid,
                "bbox": det["bbox"],
                "confidence": det.get("confidence", 0.0),
            })
    return by_track


def _median_gt_dist(dets: list[dict], gt: dict[int, tuple[float, float]]) -> float | None:
    """Median distance of detections from GT center."""
    dists = []
    for det in dets:
        fid = det["frame_id"]
        if fid not in gt:
            continue
        bbox = det["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        dists.append(math.sqrt((cx - gt[fid][0]) ** 2 + (cy - gt[fid][1]) ** 2))
    if not dists:
        return None
    dists.sort()
    mid = len(dists) // 2
    return dists[mid] if len(dists) % 2 else (dists[mid - 1] + dists[mid]) / 2


def main() -> None:
    gt = _load_gt()
    by_track = _seg_detections_by_track(ARNO_STEM)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Running Arno at threshold=0.50")
    print(sep)
    manifest_50 = _run_tracking(0.50)
    cps_50 = manifest_50.get("candidate_pool_stats") or {}
    n_det_50 = sum(1 for fr in manifest_50.get("frames", []) if fr.get("detected"))
    n_int_50 = sum(1 for fr in manifest_50.get("frames", []) if fr.get("interpolated"))

    print(f"\n{sep}")
    print("  Running Arno at threshold=0.60")
    print(sep)
    manifest_60 = _run_tracking(0.60)
    cps_60 = manifest_60.get("candidate_pool_stats") or {}
    n_det_60 = sum(1 for fr in manifest_60.get("frames", []) if fr.get("detected"))
    n_int_60 = sum(1 for fr in manifest_60.get("frames", []) if fr.get("interpolated"))

    # --- Compare ---
    incl_50 = {t["track_id"]: t for t in cps_50.get("tracks_included", [])}
    incl_60 = {t["track_id"]: t for t in cps_60.get("tracks_included", [])}
    excl_60 = {t["track_id"]: t for t in cps_60.get("tracks_excluded", [])}

    swing_track_ids = set(incl_50) - set(incl_60)

    print(f"\n{sep}")
    print("  ARNO THRESHOLD COMPARISON")
    print(sep)
    delta_det = n_det_60 - n_det_50
    delta_int = n_int_60 - n_int_50
    print(f"\n  Detected frames:     0.50={n_det_50}, 0.60={n_det_60}  (delta={delta_det:+d})")
    print(f"  Interpolated frames: 0.50={n_int_50}, 0.60={n_int_60}  (delta={delta_int:+d})")
    print(f"\n  Tracks included at 0.50: {len(incl_50)}")
    print(f"  Tracks included at 0.60: {len(incl_60)}")
    print(f"  Swing tracks (in 0.50 but not 0.60): {sorted(swing_track_ids)}")
    near_60 = cps_60.get("threshold_sensitivity_tracks", [])
    print(f"\n  Near-boundary at 0.60 (+-0.05): {near_60}")

    hdr = f"  {'Track':>8} | {'Score@0.50':>10} | {'n_dets':>6} | {'Median GT dist (px)':>20} | {'Frame range':>20}"
    print(f"\n{hdr}")
    print("  " + "-" * 75)
    for tid in sorted(swing_track_ids):
        info_50 = incl_50.get(tid) or excl_60.get(tid) or {}
        dets = by_track.get(tid, [])
        med_dist = _median_gt_dist(dets, gt)
        if dets:
            first_fid = min(d["frame_id"] for d in dets)
            last_fid = max(d["frame_id"] for d in dets)
            frame_range = f"{first_fid}-{last_fid}"
        else:
            frame_range = "N/A"
        dist_str = f"{med_dist:.1f}" if med_dist is not None else "N/A"
        score = info_50.get("score", "?")
        n_dets = info_50.get("n_detections", len(dets))
        print(f"  {tid:>8} | {score:>10} | {n_dets:>6} | {dist_str:>20} | {frame_range:>20}")

    print("\n  --- All excluded at 0.60 (for context) ---")
    print(f"  {'Track':>8} | {'Score':>10} | {'n_dets':>6} | {'Median GT dist (px)':>20}")
    print("  " + "-" * 55)
    for t in sorted(cps_60.get("tracks_excluded", []), key=lambda x: x["score"], reverse=True):
        dets = by_track.get(t["track_id"], [])
        med_dist = _median_gt_dist(dets, gt)
        dist_str = f"{med_dist:.1f}" if med_dist is not None else "N/A"
        print(f"  {t['track_id']:>8} | {t['score']:>10} | {t['n_detections']:>6} | {dist_str:>20}")


if __name__ == "__main__":
    main()
