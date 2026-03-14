"""Experiment 20 — sweep of_trace_filter_enabled=False with various OF configs.

Tests whether the Phase A OF trace contributes positively as a *pure soft score
signal* (no hard proximity gate) vs the no-OF baseline (Exp 15).

Usage:
    uv run python scripts/sweep_trace_filter.py
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
CONFIG_PATH = Path("config.yaml")
RESULTS_PATH = Path("output/sweep_trace_filter_results.json")

# (label, of_method, of_trace_filter_enabled, of_gap_fill_enabled)
CONFIGS = [
    # Exp 15 baseline (no OF at all)
    ("no-OF baseline (Exp 15)", "none", True, True),
    # Trace score only, gap-fill active (most useful: score + fill, no hard gate)
    ("OF trace score + gap-fill, no filter", "auto", False, True),
    # Trace score only, no gap-fill (pure soft signal, linear interp for gaps)
    ("OF trace score only, no gap-fill", "auto", False, False),
]


def _run_tracking(
    video_stem: str,
    of_method: str,
    of_trace_filter_enabled: bool,
    of_gap_fill_enabled: bool,
) -> None:
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
        merge_score_threshold=trk_cfg.get("merge_score_threshold", 0.6),
        merge_min_detection_conf=0.0,
        merge_threshold_adaptive=True,
        merge_threshold_low=0.5,
        merge_threshold_high=0.6,
        merge_threshold_min_overlap_ratio=0.55,
        w_continuity=0.6,
        w_track_stickiness=0.4,
        optical_flow_method=of_method,
        of_gap_fill_enabled=of_gap_fill_enabled,
        of_min_gap_for_fill=0,
        of_drift_guard_px=0.0,
        of_reanchor_min_conf=0.0,
        of_trace_filter_enabled=of_trace_filter_enabled,
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
    )


def _evaluate(video_stem: str) -> dict:
    from src.tracking.evaluate import evaluate_tracking
    tracking_dir = Path("output/tracking") / video_stem
    gt_path = Path("annotations/tracking") / video_stem / "gt_centers.csv"
    return evaluate_tracking(tracking_dir, gt_path)


def main() -> None:
    all_results: dict[str, dict] = {}

    for label, of_method, of_trace_filter_enabled, of_gap_fill_enabled in CONFIGS:
        print(f"\n{'=' * 65}")
        print(f"  CONFIG: {label}")
        print(f"{'=' * 65}")

        video_results: dict[str, object] = {}
        t0 = time.perf_counter()
        for stem, short in zip(VIDEOS, SHORT_NAMES):
            print(f"\n  --- {short} ---")
            _run_tracking(stem, of_method, of_trace_filter_enabled, of_gap_fill_enabled)
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

    # Summary table
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Config':<42} | {'Arno':>8} | {'Andreas':>8} | {'Lach':>8} | {'Mean':>8} | {'HOTA':>6} | {'Time':>5}")
    print("-" * 100)
    for label, *_ in CONFIGS:
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
        print(
            f"{label:<42} | {arno_e:>8.1f} | {andreas_e:>8.1f} | {lach_e:>8.1f} "
            f"| {mean_e:>8.1f} | {mean_h:>6.3f} | {rt:>5.0f}s"
        )


if __name__ == "__main__":
    main()
