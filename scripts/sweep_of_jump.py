"""Sweep OF + identity guard configs with merge_score_threshold=0.6.

Usage:
    uv run python scripts/sweep_of_jump.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

VIDEOS = [
    "VERBIER FREERIDE WEEK QUALIFIER 4__1_Ski Men_Arno Vuarnier_58_Switzerland_91.33",
    "VERBIER FREERIDE WEEK QUALIFIER 4__2_Ski Men_Andreas Bakke_24_Norway_89",
    "VERBIER FREERIDE WEEK QUALIFIER 4__3_Ski Men_Lach Powell_8_New Zealand_86",
]
SHORT_NAMES = ["Arno", "Andreas", "Lach"]
CONFIG_PATH = Path("config.yaml")
RESULTS_PATH = Path("output/sweep_of_jump_results.json")

# Sweep: (OF method, identity_guard_enabled, max_jump_px, merge_score_threshold)
CONFIGS = [
    # Baseline: best no-OF
    ("none", False, 100, 0.6, "no-OF / 0.6"),
    # OF + 0.6 threshold, various jump sizes
    ("auto", True, 50, 0.6, "OF / jump=50 / 0.6"),
    ("auto", True, 100, 0.6, "OF / jump=100 / 0.6"),
    ("auto", True, 150, 0.6, "OF / jump=150 / 0.6"),
    # OF + 0.6, no identity guard (OF for gap-fill only)
    ("auto", False, 100, 0.6, "OF-gapfill / 0.6"),
    # Production best for comparison
    ("auto", True, 20, 0.5, "OF / jump=20 / 0.5 (prod)"),
    # OF + 0.5 + higher jump
    ("auto", True, 50, 0.5, "OF / jump=50 / 0.5"),
]


def _run_tracking(video_stem: str, of_method: str, ig_enabled: bool, max_jump: float, merge_thresh: float) -> None:
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
        smooth_window=5,
        padding_ratio=trk_cfg.get("padding_ratio", 0.15),
        merge_tracks=True,
        merge_score_threshold=merge_thresh,
        merge_min_detection_conf=0.0,
        w_continuity=0.6,
        w_track_stickiness=0.4,
        optical_flow_method=of_method,
        flow_max_extrapolate_frames=trk_cfg.get("flow_max_extrapolate_frames", 30),
        flow_min_keypoints=trk_cfg.get("flow_min_keypoints", 5),
        identity_guard_enabled=ig_enabled,
        identity_guard_max_jump_px=max_jump,
        identity_guard_reanchor_interval=trk_cfg.get("identity_guard_reanchor_interval", 50),
        identity_guard_reanchor_min_conf=trk_cfg.get("identity_guard_reanchor_min_conf", 0.5),
        identity_guard_max_drift_px=trk_cfg.get("identity_guard_max_drift_px", 200.0),
        w_velocity=trk_cfg.get("w_velocity", 0.4),
        vel_history_len=trk_cfg.get("vel_history_len", 5),
        of_synthetic_confidence=trk_cfg.get("of_synthetic_confidence", 0.3),
        cmc_enabled=False,
        cmc_method="none",
        cmc_exclude_margin=1.5,
        cmc_min_features=20,
        cmc_ransac_threshold=3.0,
    )


def _evaluate(video_stem: str) -> dict:
    from src.tracking.evaluate import evaluate_tracking
    tracking_dir = Path("output/tracking") / video_stem
    gt_path = Path("annotations/tracking") / video_stem / "gt_centers.csv"
    return evaluate_tracking(tracking_dir, gt_path)


def main() -> None:
    all_results = {}

    for of_method, ig_enabled, max_jump, merge_thresh, label in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"  CONFIG: {label}")
        print(f"{'=' * 60}")

        video_results = {}
        t0 = time.perf_counter()
        for stem, short in zip(VIDEOS, SHORT_NAMES):
            print(f"\n  --- {short} ---")
            _run_tracking(stem, of_method, ig_enabled, max_jump, merge_thresh)
            metrics = _evaluate(stem)
            video_results[short] = metrics
            err = metrics.get("mean_error_px", "?")
            hota = metrics.get("hota", "?")
            print(f"  mean_error={err:.1f} px, HOTA={hota:.3f}")

        elapsed = time.perf_counter() - t0
        video_results["_runtime_s"] = round(elapsed, 2)
        all_results[label] = video_results

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':<30} | {'Arno':>8} | {'Andreas':>8} | {'Lach':>8} | {'Mean err':>8} | {'HOTA':>6} | {'Time':>5}")
    print("-" * 90)
    for of_method, ig_enabled, max_jump, merge_thresh, label in CONFIGS:
        r = all_results[label]
        arno_e = r["Arno"]["mean_error_px"]
        andreas_e = r["Andreas"]["mean_error_px"]
        lach_e = r["Lach"]["mean_error_px"]
        mean_e = (arno_e + andreas_e + lach_e) / 3
        arno_h = r["Arno"]["hota"]
        andreas_h = r["Andreas"]["hota"]
        lach_h = r["Lach"]["hota"]
        mean_h = (arno_h + andreas_h + lach_h) / 3
        rt = r["_runtime_s"]
        print(f"{label:<30} | {arno_e:>8.1f} | {andreas_e:>8.1f} | {lach_e:>8.1f} | {mean_e:>8.1f} | {mean_h:>6.3f} | {rt:>5.0f}s")


if __name__ == "__main__":
    main()
