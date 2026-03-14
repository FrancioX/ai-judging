"""Experiments 21a/21b — Re-test Exp 17 (conditional gap-fill) and Exp 18
(high-conf re-anchoring) with of_trace_filter_enabled=False.

All configs use the Exp 20 base: optical_flow_method=auto, of_trace_filter_enabled=false,
adaptive threshold active.

Reports average ms/frame alongside quality metrics.

Usage:
    uv run python scripts/sweep_exp21.py
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
RESULTS_PATH = Path("output/sweep_exp21_results.json")

# Each entry: (label, of_min_gap_for_fill, of_drift_guard_px, of_reanchor_min_conf)
# All configs share: optical_flow_method=auto, of_trace_filter_enabled=False,
#                    of_gap_fill_enabled=True, identity_guard_enabled=False
CONFIGS = [
    # Exp 20 winner — baseline for this sweep
    ("Exp20 baseline (soft trace, no filter)", 0, 0.0, 0.0),
    # Exp 21a — Exp 17 variant: conditional gap-fill (only gaps >= 10 frames use OF)
    ("Exp21a: conditional gap-fill (min_gap=10, drift=200)", 10, 200.0, 0.0),
    # Exp 21b — Exp 18 variant: high-conf re-anchoring
    ("Exp21b: high-conf reanchor (reanchor_min_conf=0.7)", 0, 0.0, 0.7),
    # Exp 21c — both combined
    ("Exp21c: conditional gap-fill + high-conf reanchor", 10, 200.0, 0.7),
]


def _run_tracking(
    video_stem: str,
    of_min_gap_for_fill: int,
    of_drift_guard_px: float,
    of_reanchor_min_conf: float,
) -> int:
    """Run tracking, return n_frames processed."""
    from src.pipeline import load_config
    from src.tracking.tracker import track_skier

    config = load_config(CONFIG_PATH)
    trk_cfg = config.get("tracking", {})

    seg_manifest = Path("output/segmentation") / video_stem / "segmentation.json"
    frame_dir = Path("output/frames") / video_stem
    output_dir = Path("output/tracking") / video_stem

    if not seg_manifest.exists():
        print(f"  ✗ Missing: {seg_manifest}")
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
        of_min_gap_for_fill=of_min_gap_for_fill,
        of_drift_guard_px=of_drift_guard_px,
        of_reanchor_min_conf=of_reanchor_min_conf,
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
    )

    # Read n_frames from manifest
    manifest = Path("output/tracking") / video_stem / "tracking.json"
    if manifest.exists():
        return json.loads(manifest.read_text()).get("n_frames", 0)
    return 0


def _evaluate(video_stem: str) -> dict:
    from src.tracking.evaluate import evaluate_tracking
    return evaluate_tracking(
        Path("output/tracking") / video_stem,
        Path("annotations/tracking") / video_stem / "gt_centers.csv",
    )


def main() -> None:
    all_results: dict[str, dict] = {}

    for label, of_min_gap, of_drift, of_reanchor in CONFIGS:
        print(f"\n{'=' * 68}")
        print(f"  CONFIG: {label}")
        print(f"{'=' * 68}")

        video_results: dict[str, object] = {}
        total_frames = 0
        t0 = time.perf_counter()

        for stem, short in zip(VIDEOS, SHORT_NAMES):
            print(f"\n  --- {short} ---")
            n_frames = _run_tracking(stem, of_min_gap, of_drift, of_reanchor)
            total_frames += n_frames
            metrics = _evaluate(stem)
            video_results[short] = metrics
            err = metrics.get("mean_error_px", "?")
            hota = metrics.get("hota", "?")
            print(f"  mean_error={err:.1f} px, HOTA={hota:.3f}  ({n_frames} frames)")

        elapsed = time.perf_counter() - t0
        ms_per_frame = (elapsed * 1000 / total_frames) if total_frames else 0.0
        video_results["_runtime_s"] = round(elapsed, 2)
        video_results["_total_frames"] = total_frames
        video_results["_ms_per_frame"] = round(ms_per_frame, 2)
        all_results[label] = video_results

        print(f"\n  Total: {elapsed:.1f}s | {total_frames} frames | {ms_per_frame:.1f} ms/frame")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

    # Summary table
    print(f"\n{'=' * 95}")
    print("  SUMMARY")
    print(f"{'=' * 95}")
    print(f"{'Config':<46} | {'Arno':>7} | {'Andreas':>8} | {'Lach':>6} | {'Mean err':>8} | {'HOTA':>6} | {'ms/frame':>8}")
    print("-" * 108)
    for label, *_ in CONFIGS:
        r = all_results[label]
        arno_e = r["Arno"]["mean_error_px"]
        andreas_e = r["Andreas"]["mean_error_px"]
        lach_e = r["Lach"]["mean_error_px"]
        mean_e = (arno_e + andreas_e + lach_e) / 3
        mean_h = (r["Arno"]["hota"] + r["Andreas"]["hota"] + r["Lach"]["hota"]) / 3
        ms_pf = r["_ms_per_frame"]
        print(
            f"{label:<46} | {arno_e:>7.1f} | {andreas_e:>8.1f} | {lach_e:>6.1f} "
            f"| {mean_e:>8.1f} | {mean_h:>6.3f} | {ms_pf:>7.1f}ms"
        )


if __name__ == "__main__":
    main()
