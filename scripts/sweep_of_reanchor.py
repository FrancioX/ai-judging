"""Test single-candidate-only OF trace re-anchoring on 6-video set.

Hypothesis: the OF trace re-anchoring to multiple candidates (including
bystanders) during conflict frames causes systematic wrong decisions for
Gabin Leonard (229 high-confidence conflicts). Restricting re-anchoring
to frames with exactly 1 candidate (unambiguous) prevents bystander lock-on.

Baseline: Iter 5 result (conflict_of_multiplier=1.0, conflict_stickiness_multiplier=1.0)
  Mini mean: 26.60px | Jordan: 42.6px | Gabin: 58.6px

Usage:
    uv run python scripts/sweep_of_reanchor.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

VIDEOS = [
    ("Ski Men_1_91.33_Arno Vuarnier", "Arno", 47.9, 0.734),
    ("Ski Men_2_89_Andreas Bakke", "Andreas", 8.7, 0.932),
    ("Snowboard Men_16_40_Jonatan Laland", "Jonatan", 4.6, 0.955),
    ("Snowboard Men_17_35_Quentin Puydenus", "Quentin", 45.2, 0.753),
    ("Ski Men_10_75_Jordan Koch", "Jordan", 42.6, 0.777),
    ("Ski Men_12_74_Gabin Leonard", "Gabin", 58.6, 0.869),
]

RESULTS_PATH = Path("output/sweep_of_reanchor_results.json")


def _run_tracking(video_stem: str) -> int:
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
        conflict_of_multiplier=1.0,
        conflict_stickiness_multiplier=1.0,
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
    print("Single-candidate-only OF re-anchoring (Iteration 6)")
    print("=" * 60)

    results = {}
    total_frames = 0
    t0 = time.perf_counter()

    for stem, short, b_err, b_hota in VIDEOS:
        print(f"\n--- {short} ---")
        n = _run_tracking(stem)
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

    shorts = [s for _, s, _, _ in VIDEOS]
    b_errs = {s: b for _, s, b, _ in VIDEOS}
    mini_shorts = ["Arno", "Andreas", "Jonatan", "Quentin"]

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"  {'Video':<10} {'Baseline':>9} {'New':>9} {'Δ':>6}   HOTA")
    print("-" * 50)
    for short, _, b_err, b_hota in VIDEOS:
        ne = results[short]["mean_error_px"]
        nh = results[short]["hota"]
        print(f"  {short:<10} {b_err:>9.1f} {ne:>9.1f} {ne-b_err:>+6.1f}   {nh:.3f}")

    mini_mean_new = sum(results[s]["mean_error_px"] for s in mini_shorts) / 4
    mini_mean_old = sum(b_errs[s] for s in mini_shorts) / 4
    print(f"\n  Mini mean: {mini_mean_old:.2f} → {mini_mean_new:.2f} px (Δ{mini_mean_new - mini_mean_old:+.2f})")
    print(f"  {elapsed:.0f}s | {total_frames} frames | {ms_pf:.1f} ms/frame")


if __name__ == "__main__":
    main()
