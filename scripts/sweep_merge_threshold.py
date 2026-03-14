"""Sweep merge_score_threshold values and evaluate tracking accuracy.

Usage:
    uv run python scripts/sweep_merge_threshold.py

Writes results to output/sweep_merge_threshold_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

VIDEOS = [
    "Ski Men_1_91.33_Arno Vuarnier",
    "Ski Men_2_89_Andreas Bakke",
    "Ski Men_3_86_Lach Powell",
]
SHORT_NAMES = ["Arno", "Andreas", "Lach"]
SWEEP_VALUES = [0.65, 0.70]
CONFIG_PATH = Path("config.yaml")
RESULTS_PATH = Path("output/sweep_merge_threshold_results.json")


def _run_tracking_stage(video_stem: str, merge_score_threshold: float) -> None:
    """Run tracking for one video with a specific merge_score_threshold."""
    from src.pipeline import load_config
    from src.tracking.tracker import track_skier

    config = load_config(CONFIG_PATH)
    trk_cfg = config.get("tracking", {})

    seg_dir = Path("output/segmentation") / video_stem
    seg_manifest = seg_dir / "segmentation.json"
    frame_dir = Path("output/frames") / video_stem
    output_dir = Path("output/tracking") / video_stem

    if not seg_manifest.exists():
        print(f"  ✗ Segmentation manifest not found: {seg_manifest}")
        return

    track_skier(
        seg_manifest_path=seg_manifest,
        frame_dir=frame_dir,
        output_dir=output_dir,
        w_conf=trk_cfg.get("w_conf", 0.3),
        w_center=trk_cfg.get("w_center", 0.5),
        w_length=trk_cfg.get("w_length", 0.2),
        min_track_frames=trk_cfg.get("min_track_frames", 10),
        smooth_window=trk_cfg.get("smooth_window", 5),
        padding_ratio=trk_cfg.get("padding_ratio", 0.15),
        merge_tracks=trk_cfg.get("merge_tracks", True),
        merge_score_threshold=merge_score_threshold,
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


def _evaluate(video_stem: str) -> dict:
    """Run evaluation for one video and return metrics."""
    from src.tracking.evaluate import evaluate_tracking

    tracking_dir = Path("output/tracking") / video_stem
    gt_path = Path("annotations/tracking") / video_stem / "gt_centers.csv"
    return evaluate_tracking(tracking_dir, gt_path)


def main() -> None:
    all_results: dict[str, dict] = {}

    for val in SWEEP_VALUES:
        print(f"\n{'='*60}")
        print(f"  SWEEP: merge_score_threshold = {val}")
        print(f"{'='*60}")

        video_results: dict[str, dict] = {}
        t0 = time.perf_counter()
        for stem, short in zip(VIDEOS, SHORT_NAMES):
            print(f"\n  --- {short} ---")
            _run_tracking_stage(stem, val)
            metrics = _evaluate(stem)
            video_results[short] = metrics
            mean_err = metrics.get('mean_error_px', '?')
            hota = metrics.get('hota_score', '?')
            mean_err_str = f"{mean_err:.1f}" if isinstance(mean_err, (int, float)) else str(mean_err)
            hota_str = f"{hota:.3f}" if isinstance(hota, (int, float)) else str(hota)
            print(f"  mean_error={mean_err_str} px, HOTA={hota_str}")

        elapsed = time.perf_counter() - t0
        video_results["_runtime_s"] = round(elapsed, 2)
        all_results[str(val)] = video_results

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {RESULTS_PATH}")

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Threshold':>10} | {'Arno err':>10} | {'Andreas err':>12} | {'Lach err':>10} | {'Mean HOTA':>10}")
    print("-" * 65)
    for val in SWEEP_VALUES:
        r = all_results[str(val)]
        arno_err = r.get("Arno", {}).get("mean_error_px", "?")
        andreas_err = r.get("Andreas", {}).get("mean_error_px", "?")
        lach_err = r.get("Lach", {}).get("mean_error_px", "?")
        arno_h = r.get("Arno", {}).get("hota_score", 0)
        andreas_h = r.get("Andreas", {}).get("hota_score", 0)
        lach_h = r.get("Lach", {}).get("hota_score", 0)
        mean_hota = (arno_h + andreas_h + lach_h) / 3 if all(isinstance(x, (int, float)) for x in [arno_h, andreas_h, lach_h]) else "?"
        print(f"{val:>10} | {arno_err:>10.1f} | {andreas_err:>12.1f} | {lach_err:>10.1f} | {mean_hota:>10.3f}")


if __name__ == "__main__":
    main()
