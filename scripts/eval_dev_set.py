"""Evaluate the 10-video dev set with a given tracking config.

Usage (from worktree root):
    uv run python scripts/eval_dev_set.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

DEV_VIDEOS = [
    ("Ski Men_2_89_Andreas Bakke", "Andreas", 8.7, 0.932),
    ("Ski Men_3_86_Lach Powell", "Lach", 9.2, 0.931),
    ("Ski Men_11_74.83_Emile Peizerat", "Emile", 15.7, 0.858),
    ("Ski Men_10_75_Jordan Koch", "Jordan", 42.6, 0.777),
    ("Ski Men_12_74_Gabin Leonard", "Gabin", 58.5, 0.867),
    ("Snowboard Men_16_40_Jonatan Laland", "Jonatan", 4.6, 0.955),
    ("Snowboard Men_12_52.33_Adriano Cardillo", "Adriano", 9.8, 0.913),
    ("Snowboard Men_10_54.33_Cedric Giraudeau", "Cedric", 13.1, 0.906),
    ("Snowboard Men_17_35_Quentin Puydenus", "Quentin", 47.8, 0.753),
    ("Snowboard Men_15_45_Theodor Salen", "Theodor", 54.6, 0.777),
]

RESULTS_PATH = Path("output/eval_dev_set_results.json")


def _run_tracking(video_stem: str, conflict_of_multiplier: float, conflict_stickiness_multiplier: float) -> int:
    from src.pipeline import load_config
    from src.tracking.tracker import track_skier

    config = load_config(Path("config.yaml"))
    trk_cfg = config.get("tracking", {})

    seg_manifest = Path("output/segmentation") / video_stem / "segmentation.json"
    frame_dir = Path("output/frames") / video_stem
    output_dir = Path("output/tracking") / video_stem

    if not seg_manifest.exists():
        print(f"  ✗ Missing segmentation: {seg_manifest}")
        return 0

    track_skier(
        seg_manifest_path=seg_manifest,
        frame_dir=frame_dir,
        output_dir=output_dir,
        w_conf=trk_cfg.get("w_conf", 0.3),
        w_center=trk_cfg.get("w_center", 0.5),
        w_length=trk_cfg.get("w_length", 0.2),
        min_track_frames=trk_cfg.get("min_track_frames", 10),
        smooth_window=5,
        padding_ratio=trk_cfg.get("padding_ratio", 0.15),
        merge_tracks=True,
        merge_score_threshold=trk_cfg.get("merge_score_threshold", 0.6),
        merge_min_detection_conf=0.0,
        merge_threshold_adaptive=True,
        merge_threshold_low=0.5,
        merge_threshold_high=0.6,
        merge_threshold_min_overlap_ratio=0.55,
        w_continuity=0.6,
        w_track_stickiness=0.4,
        optical_flow_method="auto",
        of_gap_fill_enabled=True,
        of_min_gap_for_fill=10,
        of_drift_guard_px=200.0,
        of_reanchor_min_conf=0.0,
        of_trace_filter_enabled=False,
        flow_max_extrapolate_frames=trk_cfg.get("flow_max_extrapolate_frames", 30),
        flow_min_keypoints=trk_cfg.get("flow_min_keypoints", 5),
        identity_guard_enabled=False,
        identity_guard_max_jump_px=100.0,
        identity_guard_reanchor_interval=50,
        identity_guard_reanchor_min_conf=0.5,
        identity_guard_max_drift_px=200.0,
        w_velocity=trk_cfg.get("w_velocity", 0.4),
        vel_history_len=trk_cfg.get("vel_history_len", 5),
        of_synthetic_confidence=trk_cfg.get("of_synthetic_confidence", 0.3),
        cmc_enabled=False,
        cmc_method="none",
        cmc_exclude_margin=1.5,
        cmc_min_features=20,
        cmc_ransac_threshold=3.0,
        cotracker_enabled=False,
        conflict_of_multiplier=conflict_of_multiplier,
        conflict_stickiness_multiplier=conflict_stickiness_multiplier,
    )

    manifest = Path("output/tracking") / video_stem / "tracking.json"
    if manifest.exists():
        try:
            return json.loads(manifest.read_text()).get("n_frames", 0)
        except json.JSONDecodeError:
            return 0
    return 0


def _evaluate(video_stem: str) -> dict:
    from src.tracking.evaluate import evaluate_tracking
    return evaluate_tracking(
        Path("output/tracking") / video_stem,
        Path("annotations/tracking") / video_stem / "gt_centers.csv",
    )


def main() -> None:
    print("10-video dev set: OF×1.0 sticky×1.0 (conflict_of_multiplier=1.0, conflict_stickiness_multiplier=1.0)")
    print("=" * 70)

    results = {}
    total_frames = 0
    t0 = time.perf_counter()

    for stem, short, b_err, b_hota in DEV_VIDEOS:
        print(f"\n--- {short} ---")
        n = _run_tracking(stem, conflict_of_multiplier=1.0, conflict_stickiness_multiplier=1.0)
        total_frames += n
        metrics = _evaluate(stem)
        results[short] = metrics
        err = metrics.get("mean_error_px", float("nan"))
        hota = metrics.get("hota", float("nan"))
        print(f"  err={err:.1f} px (Δ{err - b_err:+.1f})  HOTA={hota:.3f} (Δ{hota - b_hota:+.3f})")

    elapsed = time.perf_counter() - t0
    ms_pf = (elapsed * 1000 / total_frames) if total_frames else 0.0

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 70}")
    print("SUMMARY (Exp 21a baseline)")
    print(f"{'=' * 70}")
    shorts = [s for _, s, _, _ in DEV_VIDEOS]
    b_errs = [b for _, _, b, _ in DEV_VIDEOS]
    b_hotas = [b for _, _, _, b in DEV_VIDEOS]
    new_errs = [results[s]["mean_error_px"] for s in shorts]
    new_hotas = [results[s]["hota"] for s in shorts]

    for s, be, bh, ne, nh in zip(shorts, b_errs, b_hotas, new_errs, new_hotas):
        print(f"  {s:<10} {be:>6.1f} → {ne:>6.1f} px (Δ{ne-be:+.1f})   HOTA {bh:.3f} → {nh:.3f} (Δ{nh-bh:+.3f})")

    b_mean = sum(b_errs) / len(b_errs)
    n_mean = sum(new_errs) / len(new_errs)
    b_hota_mean = sum(b_hotas) / len(b_hotas)
    n_hota_mean = sum(new_hotas) / len(new_hotas)
    print(f"\n  {'MEAN':<10} {b_mean:>6.1f} → {n_mean:>6.1f} px (Δ{n_mean-b_mean:+.1f})   HOTA {b_hota_mean:.3f} → {n_hota_mean:.3f} (Δ{n_hota_mean-b_hota_mean:+.3f})")
    print(f"\n  {elapsed:.0f}s | {total_frames} frames | {ms_pf:.1f} ms/frame")


if __name__ == "__main__":
    main()
