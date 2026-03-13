"""Run tracking on all videos that have ground-truth annotations, then evaluate.

Uses existing segmentation manifests — no YOLO re-run.
Reports ms/frame per video and overall aggregate.

Usage:
    uv run python scripts/batch_track_and_eval.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

CONFIG_PATH = Path("config.yaml")
ANNOTATIONS_DIR = Path("annotations/tracking")
SEG_BASE = Path("output/segmentation")
FRAMES_BASE = Path("output/frames")
TRACKING_BASE = Path("output/tracking")


def _run_tracking(stem: str, trk_cfg: dict) -> tuple[float, int]:
    """Run tracking for one video. Returns (elapsed_s, n_frames)."""
    from src.tracking.tracker import track_skier

    seg_manifest = SEG_BASE / stem / "segmentation.json"
    frame_dir = FRAMES_BASE / stem
    output_dir = TRACKING_BASE / stem

    if not seg_manifest.exists():
        print(f"  ✗ Missing segmentation: {stem}")
        return 0.0, 0

    t0 = time.perf_counter()
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
        merge_score_threshold=trk_cfg.get("merge_score_threshold", 0.6),
        merge_min_detection_conf=trk_cfg.get("merge_min_detection_conf", 0.0),
        merge_threshold_adaptive=trk_cfg.get("merge_threshold_adaptive", False),
        merge_threshold_low=trk_cfg.get("merge_threshold_low", 0.5),
        merge_threshold_high=trk_cfg.get("merge_threshold_high", 0.6),
        merge_threshold_min_overlap_ratio=trk_cfg.get("merge_threshold_min_overlap_ratio", 0.55),
        w_continuity=trk_cfg.get("w_continuity", 0.6),
        w_track_stickiness=trk_cfg.get("w_track_stickiness", 0.4),
        optical_flow_method=trk_cfg.get("optical_flow_method", "none"),
        of_gap_fill_enabled=trk_cfg.get("of_gap_fill_enabled", True),
        of_min_gap_for_fill=trk_cfg.get("of_min_gap_for_fill", 0),
        of_drift_guard_px=trk_cfg.get("of_drift_guard_px", 0.0),
        of_reanchor_min_conf=trk_cfg.get("of_reanchor_min_conf", 0.0),
        of_trace_filter_enabled=trk_cfg.get("of_trace_filter_enabled", True),
        flow_max_extrapolate_frames=trk_cfg.get("flow_max_extrapolate_frames", 30),
        flow_min_keypoints=trk_cfg.get("flow_min_keypoints", 5),
        identity_guard_enabled=trk_cfg.get("identity_guard_enabled", False),
        identity_guard_max_jump_px=trk_cfg.get("identity_guard_max_jump_px", 100.0),
        identity_guard_reanchor_interval=trk_cfg.get("identity_guard_reanchor_interval", 50),
        identity_guard_reanchor_min_conf=trk_cfg.get("identity_guard_reanchor_min_conf", 0.5),
        identity_guard_max_drift_px=trk_cfg.get("identity_guard_max_drift_px", 200.0),
        w_velocity=trk_cfg.get("w_velocity", 0.4),
        vel_history_len=trk_cfg.get("vel_history_len", 5),
        of_synthetic_confidence=trk_cfg.get("of_synthetic_confidence", 0.3),
        cmc_enabled=trk_cfg.get("cmc_enabled", False),
        cmc_method=trk_cfg.get("cmc_method", "none"),
        cmc_exclude_margin=trk_cfg.get("cmc_exclude_margin", 1.5),
        cmc_min_features=trk_cfg.get("cmc_min_features", 20),
        cmc_ransac_threshold=trk_cfg.get("cmc_ransac_threshold", 3.0),
    )
    elapsed = time.perf_counter() - t0

    manifest_path = TRACKING_BASE / stem / "tracking.json"
    n_frames = 0
    if manifest_path.exists():
        n_frames = json.loads(manifest_path.read_text()).get("n_frames", 0)

    return elapsed, n_frames


def _evaluate(stem: str) -> dict:
    from src.tracking.evaluate import evaluate_tracking
    return evaluate_tracking(
        TRACKING_BASE / stem,
        ANNOTATIONS_DIR / stem / "gt_centers.csv",
    )


def main() -> None:
    from src.pipeline import load_config

    config = load_config(CONFIG_PATH)
    trk_cfg = config.get("tracking", {})

    print(f"Config: optical_flow_method={trk_cfg.get('optical_flow_method')}, "
          f"of_trace_filter_enabled={trk_cfg.get('of_trace_filter_enabled')}, "
          f"of_min_gap_for_fill={trk_cfg.get('of_min_gap_for_fill')}, "
          f"of_drift_guard_px={trk_cfg.get('of_drift_guard_px')}")

    # Only run videos that have GT annotations
    stems = sorted(p.parent.name for p in ANNOTATIONS_DIR.glob("*/gt_centers.csv"))
    print(f"Running tracking on {len(stems)} annotated videos...\n")

    results = []
    total_frames = 0
    total_time = 0.0

    for stem in stems:
        parts = stem.split("_")
        short = parts[8] if len(parts) > 8 else stem[:30]
        print(f"--- {short} ---")
        elapsed, n_frames = _run_tracking(stem, trk_cfg)
        metrics = _evaluate(stem)
        total_frames += n_frames
        total_time += elapsed
        ms_pf = (elapsed * 1000 / n_frames) if n_frames else 0.0
        err = metrics.get("mean_error_px", 0.0)
        hota = metrics.get("hota", 0.0)
        print(f"  {err:.1f} px | HOTA {hota:.3f} | {n_frames} frames | {elapsed:.1f}s | {ms_pf:.1f} ms/frame\n")
        results.append({"stem": stem, "metrics": metrics, "elapsed_s": elapsed,
                        "n_frames": n_frames, "ms_per_frame": ms_pf})

    mean_err = sum(r["metrics"]["mean_error_px"] for r in results) / len(results)
    mean_hota = sum(r["metrics"]["hota"] for r in results) / len(results)
    overall_ms_pf = (total_time * 1000 / total_frames) if total_frames else 0.0

    print("=" * 60)
    print(f"  Videos:          {len(results)}")
    print(f"  Mean error (px): {mean_err:.1f}")
    print(f"  Mean HOTA:       {mean_hota:.3f}")
    print(f"  Total frames:    {total_frames}")
    print(f"  Total time (s):  {total_time:.1f}")
    print(f"  ms/frame:        {overall_ms_pf:.1f}")
    print("=" * 60)

    out = Path("output/experiments/batch_track_eval_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {k: trk_cfg.get(k) for k in [
            "optical_flow_method", "of_trace_filter_enabled", "of_min_gap_for_fill",
            "of_drift_guard_px", "merge_threshold_adaptive",
        ]},
        "summary": {"mean_error_px": mean_err, "mean_hota": mean_hota,
                    "ms_per_frame": overall_ms_pf, "total_frames": total_frames},
        "videos": results,
    }, indent=2))


if __name__ == "__main__":
    main()
