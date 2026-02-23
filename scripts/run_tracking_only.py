"""Run only tracking + post-tracking visualization on existing segmentation output."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.tracking.tracker import track_skier
from src.visualization.render import visualize_tracking_boxes
from src.utils.video import load_frame_meta

STEM = "VERBIER FREERIDE WEEK QUALIFIER 4__1_Ski Men_Arno Vuarnier_58_Switzerland_91.33"

with open("config.yaml") as f:
    config = yaml.safe_load(f)

frame_dir = Path("output/frames") / STEM
seg_manifest = Path("output/segmentation") / STEM / "segmentation.json"
track_dir = Path("output/tracking") / STEM
viz_dir = Path("output/visualizations") / STEM

trk_cfg = config.get("tracking", {})
seg_cfg = config.get("segmentation", {})

print("=== Tracking (with optical flow) ===")
track_manifest = track_skier(
    seg_manifest,
    frame_dir,
    track_dir,
    w_conf=trk_cfg.get("w_conf", 0.3),
    w_center=trk_cfg.get("w_center", 0.5),
    w_length=trk_cfg.get("w_length", 0.2),
    min_track_frames=trk_cfg.get("min_track_frames", 10),
    smooth_window=trk_cfg.get("smooth_window", 5),
    padding_ratio=seg_cfg.get("padding_ratio", 0.15),
    merge_tracks=trk_cfg.get("merge_tracks", True),
    merge_top_n=trk_cfg.get("merge_top_n", 5),
    merge_score_threshold=trk_cfg.get("merge_score_threshold", 0.3),
    optical_flow_method=trk_cfg.get("optical_flow_method", "auto"),
    flow_max_extrapolate_frames=trk_cfg.get("flow_max_extrapolate_frames", 30),
    flow_min_keypoints=trk_cfg.get("flow_min_keypoints", 5),
)

print("\n=== Post-Tracking Visualization ===")
meta = load_frame_meta(frame_dir)
viz_fps = meta.get("effective_fps", 30)
visualize_tracking_boxes(frame_dir, track_manifest, viz_dir, fps=viz_fps)

print("\nDone!")
