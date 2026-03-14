"""Sweep Phase A conflict-resolution multipliers on the 6-video set.

The 4-video mini set (Arno/Quentin have 0 Phase A conflicts — no effect there)
plus Jordan Koch (47 conflicts, 0.32 margin) and Gabin Leonard (229 conflicts,
0.99 margin), which are the two videos most likely to respond to this change.

The hypothesis: the current OF multiplier of 3.0 during genuine conflicts
causes the tracker to lock onto whatever the OF trace is following (which
may be a bystander rather than the main skier). Reducing it should let
stickiness and quality play a larger role.

Usage (from worktree root):
    uv run python scripts/sweep_conflict_weights.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

MINI_VIDEOS = [
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

CONFLICT_VIDEOS = [
    (
        "Ski Men_10_75_Jordan Koch",
        "Jordan",
    ),
    (
        "Ski Men_12_74_Gabin Leonard",
        "Gabin",
    ),
]

ALL_VIDEOS = MINI_VIDEOS + CONFLICT_VIDEOS

# (label, conflict_of_multiplier, conflict_stickiness_multiplier)
CONFIGS = [
    ("baseline (3.0/0.2)", 3.0, 0.2),
    ("OF×1.0 sticky×0.5", 1.0, 0.5),
    ("OF×1.0 sticky×1.0", 1.0, 1.0),
    ("OF×1.5 sticky×0.5", 1.5, 0.5),
]

RESULTS_PATH = Path("output/sweep_conflict_weights_results.json")

BASELINE = {
    "Arno":    {"mean_error_px": 47.9, "hota": 0.734},
    "Andreas": {"mean_error_px":  8.7, "hota": 0.932},
    "Jonatan": {"mean_error_px":  4.6, "hota": 0.955},
    "Quentin": {"mean_error_px": 47.8, "hota": 0.753},
    "Jordan":  {"mean_error_px": 42.6, "hota": 0.777},
    "Gabin":   {"mean_error_px": 58.5, "hota": 0.867},
}


def _run_tracking(
    video_stem: str,
    conflict_of_multiplier: float,
    conflict_stickiness_multiplier: float,
) -> int:
    from src.pipeline import load_config
    from src.tracking.tracker import track_skier

    config = load_config(Path("config.yaml"))
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
    all_results: dict[str, dict] = {}
    shorts = [s for _, s in ALL_VIDEOS]

    for label, of_mult, stick_mult in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"  CONFIG: {label}")
        print(f"{'=' * 60}")

        video_results: dict[str, object] = {}
        total_frames = 0
        t0 = time.perf_counter()

        for stem, short in ALL_VIDEOS:
            print(f"\n  --- {short} ---")
            n = _run_tracking(stem, of_mult, stick_mult)
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
        video_results["_ms_per_frame"] = round(ms_pf, 2)
        all_results[label] = video_results
        print(f"\n  {elapsed:.1f}s | {total_frames} frames | {ms_pf:.1f} ms/frame")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    header = f"{'Config':<22} | " + " | ".join(f"{s:>7}" for s in shorts) + " | Mean(mini) | Mean(all) | ms/fr"
    print(header)
    print("-" * len(header))

    for label, *_ in CONFIGS:
        r = all_results[label]
        errs = [r[s]["mean_error_px"] for s in shorts]
        mini_mean = sum(errs[:4]) / 4
        all_mean = sum(errs) / len(errs)
        ms_pf = r["_ms_per_frame"]
        marker = " *" if label == "baseline (3.0/0.2)" else "  "
        vals = " | ".join(f"{e:>7.1f}" for e in errs)
        print(f"{label:<22}{marker}| {vals} | {mini_mean:>10.2f} | {all_mean:>9.2f} | {ms_pf:>5.1f}ms")

    print("\n  (* = current production config)")


if __name__ == "__main__":
    main()
