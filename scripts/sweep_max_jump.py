"""Sweep max_jump_px values and evaluate tracking accuracy.

Usage:
    uv run python scripts/sweep_max_jump.py

Writes results to output/sweep_max_jump_results.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

VIDEOS = [
    "VERBIER FREERIDE WEEK QUALIFIER 4__1_Ski Men_Arno Vuarnier_58_Switzerland_91.33",
    "VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89",
    "VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86",
]
SHORT_NAMES = ["Arno", "Andreas", "Lach"]
SWEEP_VALUES = [10, 20, 50, 100]
CONFIG_PATH = Path("config.yaml")
RESULTS_PATH = Path("output/sweep_max_jump_results.json")


def _run_tracking_stage(video_stem: str, max_jump_px: float) -> None:
    """Run tracking for one video with a specific max_jump_px."""
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
        smooth_window=trk_cfg.get("smooth_window", 0),
        padding_ratio=trk_cfg.get("padding_ratio", 0.15),
        merge_tracks=trk_cfg.get("merge_tracks", True),
        merge_score_threshold=trk_cfg.get("merge_score_threshold", 0.3),
        optical_flow_method=trk_cfg.get("optical_flow_method", "auto"),
        flow_max_extrapolate_frames=trk_cfg.get("flow_max_extrapolate_frames", 30),
        flow_min_keypoints=trk_cfg.get("flow_min_keypoints", 5),
        identity_guard_enabled=True,
        identity_guard_max_jump_px=float(max_jump_px),
        identity_guard_reanchor_interval=trk_cfg.get("identity_guard_reanchor_interval", 50),
        identity_guard_reanchor_min_conf=trk_cfg.get("identity_guard_reanchor_min_conf", 0.5),
        identity_guard_max_drift_px=trk_cfg.get("identity_guard_max_drift_px", 200.0),
        w_velocity=trk_cfg.get("w_velocity", 0.4),
        vel_history_len=trk_cfg.get("vel_history_len", 5),
        of_synthetic_confidence=trk_cfg.get("of_synthetic_confidence", 0.3),
        cmc_enabled=trk_cfg.get("cmc_enabled", True),
        cmc_method=trk_cfg.get("cmc_method", "orb"),
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
        print(f"  SWEEP: max_jump_px = {val}")
        print(f"{'='*60}")

        video_results: dict[str, dict] = {}
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

            # Read jump stats from manifest
            manifest_path = Path("output/tracking") / stem / "tracking.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                js = manifest.get("identity_guard_jump_stats", {})
                video_results[short]["rejections"] = manifest.get("identity_guard_rejections", 0)
                video_results[short]["jump_stats"] = js

        all_results[str(val)] = video_results

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
