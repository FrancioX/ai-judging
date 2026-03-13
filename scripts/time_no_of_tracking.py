"""Measure tracking time for no-OF config (Exp 15 equivalent) across all 18
annotated videos. Writes tracking output to a temp directory so it doesn't
overwrite the current production Exp21a results.

Usage:
    uv run python scripts/time_no_of_tracking.py
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

CONFIG_PATH = Path("config.yaml")
ANNOTATIONS_DIR = Path("annotations/tracking")
SEG_BASE = Path("output/segmentation")
FRAMES_BASE = Path("output/frames")


def _run_tracking_timed(stem: str, trk_cfg: dict, output_dir: Path) -> tuple[float, int]:
    from src.tracking.tracker import track_skier

    seg_manifest = SEG_BASE / stem / "segmentation.json"
    frame_dir = FRAMES_BASE / stem

    if not seg_manifest.exists():
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
        # Force no-OF regardless of current config
        optical_flow_method="none",
        of_gap_fill_enabled=True,
        of_min_gap_for_fill=0,
        of_drift_guard_px=0.0,
        of_reanchor_min_conf=0.0,
        of_trace_filter_enabled=True,
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
    elapsed = time.perf_counter() - t0

    manifest = output_dir / "tracking.json"
    n_frames = json.loads(manifest.read_text()).get("n_frames", 0) if manifest.exists() else 0
    return elapsed, n_frames


def main() -> None:
    from src.pipeline import load_config

    config = load_config(CONFIG_PATH)
    trk_cfg = config.get("tracking", {})

    stems = sorted(p.parent.name for p in ANNOTATIONS_DIR.glob("*/gt_centers.csv"))
    print(f"Timing no-OF tracking on {len(stems)} annotated videos (temp output dir)...\n")

    rows = []
    with tempfile.TemporaryDirectory(prefix="noOF_timing_") as tmp:
        tmp_path = Path(tmp)
        for stem in stems:
            out_dir = tmp_path / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            elapsed, n_frames = _run_tracking_timed(stem, trk_cfg, out_dir)
            ms_pf = elapsed * 1000 / n_frames if n_frames else 0.0
            short = stem[40:70]  # readable slice of stem
            print(f"  {short}: {elapsed:.1f}s | {n_frames} frames | {ms_pf:.1f} ms/frame")
            rows.append((stem, elapsed, n_frames, ms_pf))

    total_t = sum(r[1] for r in rows)
    total_f = sum(r[2] for r in rows)
    overall_ms_pf = total_t * 1000 / total_f if total_f else 0.0

    print(f"\n{'=' * 55}")
    print(f"  Config:          no-OF (Exp 15 equivalent)")
    print(f"  Videos:          {len(rows)}")
    print(f"  Total frames:    {total_f}")
    print(f"  Total time (s):  {total_t:.1f}")
    print(f"  ms/frame:        {overall_ms_pf:.2f}")
    print(f"{'=' * 55}")

    out = Path("output/experiments/no_of_timing.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": "no-OF (Exp 15 equivalent)",
        "total_frames": total_f,
        "total_time_s": round(total_t, 2),
        "ms_per_frame": round(overall_ms_pf, 2),
        "videos": [{"stem": r[0], "elapsed_s": round(r[1], 2),
                    "n_frames": r[2], "ms_per_frame": round(r[3], 2)} for r in rows],
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
