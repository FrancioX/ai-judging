"""Test CoTracker3 Phase B.1 long-gap fill on the 4-video mini set.

CoTracker3 replaces LK optical flow for gaps >= cotracker_min_gap.
This sweep tests: disabled vs enabled (min_gap 20, 30).

Usage (from worktree root):
    uv run python scripts/sweep_cotracker.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

VIDEOS = [
    (
        "Ski Men_1_91.33_Arno Vuarnier",
        "Arno",
    ),
    (
        "Ski Men_2_89_Andreas Bakke",
        "Andreas",
    ),
    (
        "Snowboard Men_16_40_Jonatan Laland",
        "Jonatan",
    ),
    (
        "Snowboard Men_17_35_Quentin Puydenus",
        "Quentin",
    ),
]

# (label, cotracker_enabled, cotracker_min_gap)
CONFIGS = [
    ("no CoTracker (baseline)", False, 20),
    ("CoTracker min_gap=20", True, 20),
    ("CoTracker min_gap=30", True, 30),
]

CONFIG_PATH = Path("config.yaml")
RESULTS_PATH = Path("output/sweep_cotracker_results.json")

BASELINE = {
    "Arno": {"mean_error_px": 47.9, "hota": 0.734},
    "Andreas": {"mean_error_px": 8.7, "hota": 0.932},
    "Jonatan": {"mean_error_px": 4.6, "hota": 0.955},
    "Quentin": {"mean_error_px": 47.8, "hota": 0.753},
}


def _run_tracking(video_stem: str, cotracker_enabled: bool, cotracker_min_gap: int) -> int:
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
        cotracker_enabled=cotracker_enabled,
        cotracker_min_gap=cotracker_min_gap,
        cotracker_resize_h=320,
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
    all_results: dict[str, dict] = {}

    for label, cotracker_enabled, cotracker_min_gap in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"  CONFIG: {label}")
        print(f"{'=' * 60}")

        video_results: dict[str, object] = {}
        total_frames = 0
        t0 = time.perf_counter()

        for stem, short in VIDEOS:
            print(f"\n  --- {short} ---")
            n = _run_tracking(stem, cotracker_enabled, cotracker_min_gap)
            total_frames += n
            metrics = _evaluate(stem)
            video_results[short] = metrics
            err = metrics.get("mean_error_px", float("nan"))
            hota = metrics.get("hota", float("nan"))
            b_err = BASELINE[short]["mean_error_px"]
            b_hota = BASELINE[short]["hota"]
            print(
                f"  err={err:.1f} px (Δ{err - b_err:+.1f})  "
                f"HOTA={hota:.3f} (Δ{hota - b_hota:+.3f})"
            )

        elapsed = time.perf_counter() - t0
        ms_pf = (elapsed * 1000 / total_frames) if total_frames else 0.0
        video_results["_runtime_s"] = round(elapsed, 2)
        video_results["_total_frames"] = total_frames
        video_results["_ms_per_frame"] = round(ms_pf, 2)
        all_results[label] = video_results
        print(f"\n  {elapsed:.1f}s | {total_frames} frames | {ms_pf:.1f} ms/frame")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

    print(f"\n{'=' * 80}")
    print("  SUMMARY  (baseline: Arno=47.9, Andreas=8.7, Jonatan=4.6, Quentin=47.8)")
    print(f"{'=' * 80}")
    print(
        f"{'Config':<28} | {'Arno':>8} | {'Andreas':>8} | "
        f"{'Jonatan':>8} | {'Quentin':>8} | {'Mean err':>9} | {'HOTA':>6} | {'ms/fr':>6}"
    )
    print("-" * 100)

    for label, *_ in CONFIGS:
        r = all_results[label]
        errs = [r[s]["mean_error_px"] for s in ("Arno", "Andreas", "Jonatan", "Quentin")]
        hotas = [r[s]["hota"] for s in ("Arno", "Andreas", "Jonatan", "Quentin")]
        mean_e = sum(errs) / len(errs)
        mean_h = sum(hotas) / len(hotas)
        ms_pf = r["_ms_per_frame"]
        marker = " *" if label == "no CoTracker (baseline)" else "  "
        print(
            f"{label:<28}{marker}| {errs[0]:>8.1f} | {errs[1]:>8.1f} | "
            f"{errs[2]:>8.1f} | {errs[3]:>8.1f} | {mean_e:>9.2f} | {mean_h:>6.3f} | {ms_pf:>5.1f}ms"
        )

    print("\n  (* = current production config)")


if __name__ == "__main__":
    main()
