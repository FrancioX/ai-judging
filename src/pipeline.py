"""Main pipeline: video → segmentation (YOLO-Seg) → 2D pose on crops (YOLO-Pose) → visualization.

Quality-first 7-stage pipeline:
1. Frame extraction
2. Person segmentation + crop extraction (YOLO-Seg)
3. 2D pose estimation from crops with coordinate remapping (YOLO-Pose)
4. Visualization (Pre-Tracking)
5. Temporal Tracking
6. 2D Pose Overlay Visualization
7. Visualization (Post-Tracking)

Disabled stages (re-enable when models are ready):
- 3D pose lifting (MotionBERT)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.utils.video import extract_frames, load_frame_meta
from src.segmentation.yolo_seg import segment_skier
from src.tracking.tracker import track_skier
from src.pose_2d.yolo_pose import estimate_2d_poses as estimate_2d_poses_yolo
from src.pose_3d.lifter import lift_to_3d
from src.visualization.render import (
    visualize_2d_overlay,
    visualize_3d_plotly,
    visualize_3d_side_by_side,
    visualize_segmentation_boxes,
    visualize_tracking_boxes,
)


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load pipeline configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


# ── STAGES REGISTRY ────────────────────────────────────────────────────────
# Metadata for each pipeline stage: name, function, upstream dependencies,
# and output directory within output/<stage>/<stem>/
STAGES = {
    "frames": {
        "module": "src.utils.video",
        "function": "extract_frames",
        "display_name": "Frame Extraction",
        "dependencies": [],  # Needs video file on disk
        "output_dir": "frames",
    },
    "segmentation": {
        "module": "src.segmentation.yolo_seg",
        "function": "segment_skier",
        "display_name": "Person Segmentation (YOLO-Seg)",
        "dependencies": ["frames"],
        "output_dir": "segmentation",
    },
    "tracking": {
        "module": "src.tracking.tracker",
        "function": "track_skier",
        "display_name": "Temporal Tracking",
        "dependencies": ["segmentation"],
        "output_dir": "tracking",
    },
    "pose_2d": {
        "module": "src.pose_2d.yolo_pose",
        "function": "estimate_2d_poses",
        "display_name": "2D Pose Estimation (YOLO-Pose)",
        "dependencies": ["frames", "segmentation"],  # Needs either tracking or segmentation
        "output_dir": "poses_2d",
    },
    "pose_3d": {
        "module": "src.pose_3d.lifter",
        "function": "lift_to_3d",
        "display_name": "3D Pose Lifting (MotionBERT)",
        "dependencies": ["pose_2d"],
        "output_dir": "poses_3d",
    },
    "visualization": {
        "module": "src.visualization.render",
        "function": None,  # Special: multiple viz functions
        "display_name": "Visualization",
        "dependencies": ["frames", "pose_2d"],  # Also optionally segmentation/tracking
        "output_dir": "visualizations",
    },
}


def _resolve_paths(video_path: str | Path, config: dict) -> tuple[str, Path]:
    """Derive video stem and base output directory.

    Returns
    -------
    tuple[str, Path]
        (video_stem, base_out_path)
    """
    video_path = Path(video_path)
    video_stem = video_path.stem
    base_out = Path(config.get("output_dir", "output"))
    return video_stem, base_out


def _stage_config(config: dict, stage_name: str) -> dict:
    """Extract stage-specific kwargs from config.

    Parameters
    ----------
    config : dict
        Full pipeline configuration
    stage_name : str
        Stage name (e.g., 'segmentation', 'tracking')

    Returns
    -------
    dict
        Kwargs to pass to the stage function
    """
    if stage_name == "frames":
        fe_cfg = config.get("frame_extraction", {})
        return {
            "fps": fe_cfg.get("fps"),
            "max_frames": fe_cfg.get("max_frames"),
            "fmt": fe_cfg.get("format", "jpg"),
            "quality": fe_cfg.get("quality", 95),
            "trim_start_seconds": fe_cfg.get("trim_start_seconds", 0),
            "trim_end_seconds": fe_cfg.get("trim_end_seconds", 0),
        }
    elif stage_name == "segmentation":
        seg_cfg = config.get("segmentation", {})
        return {
            "model_name": seg_cfg.get("model", "yolo11x-seg"),
            "device": seg_cfg.get("device", "mps"),
            "confidence": seg_cfg.get("confidence", 0.5),
            "padding_ratio": seg_cfg.get("padding_ratio", 0.15),
            "select_strategy": seg_cfg.get("select_strategy", "center"),
            "imgsz": seg_cfg.get("imgsz", 640),
        }
    elif stage_name == "tracking":
        trk_cfg = config.get("tracking", {})
        seg_cfg = config.get("segmentation", {})
        return {
            "w_conf": trk_cfg.get("w_conf", 0.3),
            "w_center": trk_cfg.get("w_center", 0.5),
            "w_length": trk_cfg.get("w_length", 0.2),
            "min_track_frames": trk_cfg.get("min_track_frames", 10),
            "smooth_window": trk_cfg.get("smooth_window", 5),
            "padding_ratio": seg_cfg.get("padding_ratio", 0.15),
            "merge_tracks": trk_cfg.get("merge_tracks", True),
            "merge_score_threshold": trk_cfg.get("merge_score_threshold", 0.3),
            "merge_min_detection_conf": trk_cfg.get("merge_min_detection_conf", 0.0),
            "merge_threshold_adaptive": trk_cfg.get("merge_threshold_adaptive", False),
            "merge_threshold_low": trk_cfg.get("merge_threshold_low", 0.5),
            "merge_threshold_high": trk_cfg.get("merge_threshold_high", 0.6),
            "merge_threshold_min_overlap_ratio": trk_cfg.get("merge_threshold_min_overlap_ratio", 0.3),
            "of_gap_fill_enabled": trk_cfg.get("of_gap_fill_enabled", True),
            "of_min_gap_for_fill": trk_cfg.get("of_min_gap_for_fill", 0),
            "of_drift_guard_px": trk_cfg.get("of_drift_guard_px", 0.0),
            "of_reanchor_min_conf": trk_cfg.get("of_reanchor_min_conf", 0.0),
            "of_trace_filter_enabled": trk_cfg.get("of_trace_filter_enabled", True),
            "w_continuity": trk_cfg.get("w_continuity", 0.6),
            "w_track_stickiness": trk_cfg.get("w_track_stickiness", 0.4),
            "optical_flow_method": trk_cfg.get("optical_flow_method", "auto"),
            "flow_max_extrapolate_frames": trk_cfg.get("flow_max_extrapolate_frames", 30),
            "flow_min_keypoints": trk_cfg.get("flow_min_keypoints", 5),
            "identity_guard_enabled": trk_cfg.get("identity_guard_enabled", True),
            "identity_guard_max_jump_px": trk_cfg.get("identity_guard_max_jump_px", 150.0),
            "identity_guard_reanchor_interval": trk_cfg.get("identity_guard_reanchor_interval", 50),
            "identity_guard_reanchor_min_conf": trk_cfg.get("identity_guard_reanchor_min_conf", 0.5),
            "identity_guard_max_drift_px": trk_cfg.get("identity_guard_max_drift_px", 200.0),
            "w_velocity": trk_cfg.get("w_velocity", 0.4),
            "vel_history_len": trk_cfg.get("vel_history_len", 5),
            "of_synthetic_confidence": trk_cfg.get("of_synthetic_confidence", 0.3),
            "cmc_enabled": trk_cfg.get("cmc_enabled", True),
            "cmc_method": trk_cfg.get("cmc_method", "orb"),
            "cmc_exclude_margin": trk_cfg.get("cmc_exclude_margin", 1.5),
            "cmc_min_features": trk_cfg.get("cmc_min_features", 20),
            "cmc_ransac_threshold": trk_cfg.get("cmc_ransac_threshold", 3.0),
        }
    elif stage_name == "pose_2d":
        p2d_cfg = config.get("pose_2d", {})
        return {
            "model_name": p2d_cfg.get("yolo_model", "yolo11x-pose"),
            "device": p2d_cfg.get("device", "mps"),
            "confidence": p2d_cfg.get("bbox_threshold", 0.25),
            "kpt_thr": p2d_cfg.get("keypoint_threshold", 0.3),
            "imgsz": p2d_cfg.get("imgsz", 1280),
        }
    elif stage_name == "pose_3d":
        p3d_cfg = config.get("pose_3d", {})
        return {
            "model_name": p3d_cfg.get("model", "motionbert"),
            "checkpoint_path": p3d_cfg.get("checkpoint"),
            "checkpoint_url": p3d_cfg.get("checkpoint_url"),
            "device": p3d_cfg.get("device", "mps"),
            "receptive_field": p3d_cfg.get("receptive_field", 243),
            "model_kwargs": p3d_cfg.get("model_kwargs"),
            "batch_size": p3d_cfg.get("batch_size", 16),
        }
    elif stage_name == "visualization":
        viz_cfg = config.get("visualization", {})
        return {
            "overlay_2d": viz_cfg.get("overlay_2d", True),
            "viz_fps": viz_cfg.get("fps"),
        }
    else:
        return {}


def _validate_stage_inputs(
    stage_name: str, video_stem: str, base_out: Path, video_path: Path | None = None
) -> dict[str, Path]:
    """Validate that upstream stage outputs exist for a given stage.

    Parameters
    ----------
    stage_name : str
        Name of the stage to validate inputs for
    video_stem : str
        Video stem (filename without extension)
    base_out : Path
        Base output directory (e.g., Path("output"))
    video_path : Path | None
        Path to the video file (needed for 'frames' stage and FPS detection)

    Returns
    -------
    dict[str, Path]
        Dictionary of resolved upstream paths (e.g., {"frame_dir": Path(...), "seg_manifest": Path(...)})

    Raises
    ------
    FileNotFoundError
        If any required upstream output is missing
    """
    resolved = {}

    if stage_name == "frames":
        if video_path is None or not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found or not provided: {video_path}\n"
                "Frame extraction requires a valid video path."
            )
        return resolved

    # For all other stages, frames must exist
    frame_dir = base_out / "frames" / video_stem
    if not frame_dir.exists():
        raise FileNotFoundError(
            f"Frame extraction output not found: {frame_dir}\n"
            f"Run `--stage frames` first."
        )
    resolved["frame_dir"] = frame_dir

    if stage_name == "segmentation":
        return resolved

    # For tracking, pose_2d, visualization, need segmentation.json
    seg_dir = base_out / "segmentation" / video_stem
    seg_manifest = seg_dir / "segmentation.json"
    if not seg_manifest.exists():
        raise FileNotFoundError(
            f"Segmentation output not found: {seg_manifest}\n"
            f"Run `--stage segmentation` first."
        )
    resolved["seg_manifest"] = seg_manifest
    resolved["seg_dir"] = seg_dir

    if stage_name == "tracking":
        return resolved

    if stage_name == "pose_2d":
        # Check for tracking.json first; fall back to segmentation.json
        track_dir = base_out / "tracking" / video_stem
        track_manifest = track_dir / "tracking.json"
        if track_manifest.exists():
            resolved["pose_manifest"] = track_manifest
            resolved["pose_manifest_source"] = "tracking"
        else:
            resolved["pose_manifest"] = seg_manifest
            resolved["pose_manifest_source"] = "segmentation"
        return resolved

    if stage_name == "pose_3d":
        poses_2d_dir = base_out / "poses_2d" / video_stem
        poses_2d_path = poses_2d_dir / "poses_2d.json"
        if not poses_2d_path.exists():
            raise FileNotFoundError(
                f"2D pose output not found: {poses_2d_path}\n"
                f"Run `--stage pose_2d` first."
            )
        resolved["poses_2d_path"] = poses_2d_path
        return resolved

    if stage_name == "visualization":
        poses_2d_dir = base_out / "poses_2d" / video_stem
        poses_2d_path = poses_2d_dir / "poses_2d.json"
        resolved["poses_2d_path"] = poses_2d_path if poses_2d_path.exists() else None

        poses_3d_dir = base_out / "poses_3d" / video_stem
        poses_3d_path = poses_3d_dir / "poses_3d.json"
        resolved["poses_3d_path"] = poses_3d_path if poses_3d_path.exists() else None

        track_dir = base_out / "tracking" / video_stem
        track_manifest = track_dir / "tracking.json"
        resolved["track_manifest"] = track_manifest if track_manifest.exists() else None

        return resolved

    return resolved


def _get_viz_fps(frame_dir: Path, video_path: Path | None = None) -> float:
    """Get visualization FPS from frames_meta.json, or fall back to video file.

    Parameters
    ----------
    frame_dir : Path
        Directory containing frames and frames_meta.json
    video_path : Path | None
        Path to video file (fallback source)

    Returns
    -------
    float
        Frames per second for visualization
    """
    meta = load_frame_meta(frame_dir)
    viz_fps = meta.get("effective_fps")
    if viz_fps is None and video_path:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(str(video_path))
        viz_fps = cap.get(_cv2.CAP_PROP_FPS)
        cap.release()
    return viz_fps or 30.0


def run_single_stage(
    stage_name: str, video_path: str | Path, config: dict | None = None
) -> None:
    """Run a single pipeline stage independently, reusing upstream outputs.

    Parameters
    ----------
    stage_name : str
        Name of the stage to run (e.g., 'tracking', 'pose_2d')
    video_path : str | Path
        Path to the input video file
    config : dict | None
        Pipeline configuration (loaded from config.yaml if not provided)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    if stage_name not in STAGES:
        print(f"Error: unknown stage '{stage_name}'")
        print(f"Available stages: {', '.join(STAGES.keys())}")
        sys.exit(1)

    if config is None:
        config = load_config()

    video_stem, base_out = _resolve_paths(video_path, config)

    # Validate upstream inputs
    try:
        resolved = _validate_stage_inputs(stage_name, video_stem, base_out, video_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print stage header
    stage_info = STAGES[stage_name]
    print(f"\n{'='*60}")
    print(f"Running: {stage_info['display_name']}")
    print(f"Video: {video_path.name}")
    print(f"{'='*60}")

    # ── Stage: Frames ──────────────────────────────────────────────────
    if stage_name == "frames":
        frame_dir = base_out / "frames" / video_stem
        kwargs = _stage_config(config, stage_name)
        extract_frames(video_path, frame_dir, **kwargs)

    # ── Stage: Segmentation ────────────────────────────────────────────
    elif stage_name == "segmentation":
        frame_dir = resolved["frame_dir"]
        seg_dir = base_out / "segmentation" / video_stem
        kwargs = _stage_config(config, stage_name)
        segment_skier(frame_dir, seg_dir, **kwargs)

    # ── Stage: Tracking ────────────────────────────────────────────────
    elif stage_name == "tracking":
        frame_dir = resolved["frame_dir"]
        seg_manifest = resolved["seg_manifest"]
        track_dir = base_out / "tracking" / video_stem
        kwargs = _stage_config(config, stage_name)
        track_skier(seg_manifest, frame_dir, track_dir, **kwargs)

    # ── Stage: 2D Pose Estimation ──────────────────────────────────────
    elif stage_name == "pose_2d":
        frame_dir = resolved["frame_dir"]
        pose_manifest = resolved["pose_manifest"]
        pose_manifest_source = resolved.get("pose_manifest_source", "segmentation")
        if pose_manifest_source == "tracking":
            try:
                with open(pose_manifest) as f:
                    json.load(f)
            except Exception:
                print("  Warning: tracking manifest is invalid JSON; falling back to segmentation manifest")
                pose_manifest = resolved["seg_manifest"]
                pose_manifest_source = "segmentation"
        pose_2d_dir = base_out / "poses_2d" / video_stem
        kwargs = _stage_config(config, stage_name)
        print(f"  Using {pose_manifest_source} manifest for pose estimation")
        estimate_2d_poses_yolo(
            frame_dir, pose_2d_dir, segmentation_manifest=pose_manifest, **kwargs
        )

    # ── Stage: 3D Pose Lifting ─────────────────────────────────────────
    elif stage_name == "pose_3d":
        poses_2d_path = resolved["poses_2d_path"]
        pose_3d_dir = base_out / "poses_3d" / video_stem
        kwargs = _stage_config(config, stage_name)
        try:
            from src.pose_3d.lifter import lift_to_3d
            lift_to_3d(poses_2d_path, pose_3d_dir, **kwargs)
        except ImportError:
            print("Error: 3D pose lifting module not available")
            sys.exit(1)

    # ── Stage: Visualization ───────────────────────────────────────────
    elif stage_name == "visualization":
        frame_dir = resolved["frame_dir"]
        poses_2d_path = resolved.get("poses_2d_path")
        poses_3d_path = resolved.get("poses_3d_path")
        track_manifest = resolved.get("track_manifest")
        seg_manifest = resolved.get("seg_manifest")
        viz_dir = base_out / "visualizations" / video_stem
        viz_cfg = config.get("visualization", {})
        render_3d = viz_cfg.get("render_3d", True)
        export_format = viz_cfg.get("export_format", "html")

        viz_fps = _get_viz_fps(frame_dir, video_path)

        viz_dir.mkdir(parents=True, exist_ok=True)

        # Render 2D overlay if poses exist
        if poses_2d_path and poses_2d_path.exists():
            print(f"  Rendering 2D pose overlay ({viz_fps:.2f} fps)")
            visualize_2d_overlay(
                frame_dir,
                poses_2d_path,
                viz_dir,
                skeleton=True,
                draw_bbox=True,
                tracking_manifest_path=track_manifest if track_manifest and track_manifest.exists() else None,
                fps=viz_fps,
            )
        else:
            print("  2D poses not found; skipping 2D overlay")

        # Render tracking boxes if tracking exists
        if track_manifest and track_manifest.exists():
            print(f"  Rendering tracking boxes ({viz_fps:.2f} fps)")
            visualize_tracking_boxes(frame_dir, track_manifest, viz_dir, fps=viz_fps)
        else:
            print("  Tracking output not found; skipping tracking visualization")

        # Render segmentation boxes (always possible if seg exists)
        if seg_manifest and seg_manifest.exists():
            print(f"  Rendering segmentation boxes ({viz_fps:.2f} fps)")
            visualize_segmentation_boxes(frame_dir, seg_manifest, viz_dir, fps=viz_fps)

        if render_3d and poses_3d_path and poses_3d_path.exists():
            print(f"  Rendering 3D side-by-side ({viz_fps:.2f} fps)")
            visualize_3d_side_by_side(frame_dir, poses_3d_path, viz_dir, fps=viz_fps)
            if export_format == "html":
                print("  Rendering interactive 3D HTML")
                visualize_3d_plotly(poses_3d_path, viz_dir, fps=int(viz_fps))
        elif render_3d:
            print("  3D poses not found; skipping 3D visualization")

    print(f"\n{'='*60}")
    print(f"✓ {stage_info['display_name']} complete for {video_path.name}")
    print(f"  Outputs in: {base_out / stage_info['output_dir'] / video_stem}")
    print(f"{'='*60}\n")


def run_pipeline(
    video_path: str | Path,
    config: dict | None = None,
    *,
    stop_after: str | None = None,
) -> None:
    """Execute the pipeline on a single video, optionally stopping early.

    Parameters
    ----------
    stop_after : str | None
        Stop after this stage and skip the rest.  Accepted values:
        ``"frames"``, ``"segmentation"``, ``"tracking"``.
        ``None`` (default) runs the full pipeline.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    if config is None:
        config = load_config()

    # Derive output paths per video
    video_stem = video_path.stem
    base_out = Path(config.get("output_dir", "output"))
    frame_dir = base_out / "frames" / video_stem
    seg_dir = base_out / "segmentation" / video_stem
    track_dir = base_out / "tracking" / video_stem
    pose_2d_dir = base_out / "poses_2d" / video_stem
    pose_3d_dir = base_out / "poses_3d" / video_stem
    viz_dir = base_out / "visualizations" / video_stem

    # ── Step 1: Extract frames ──────────────────────────────────────────
    fe_cfg = config.get("frame_extraction", {})
    print(f"\n{'='*60}")
    print(f"Step 1/8 - Frame Extraction: {video_path.name}")
    print(f"{'='*60}")
    extract_frames(
        video_path,
        frame_dir,
        fps=fe_cfg.get("fps"),
        max_frames=fe_cfg.get("max_frames"),
        fmt=fe_cfg.get("format", "jpg"),
        quality=fe_cfg.get("quality", 95),
        trim_start_seconds=fe_cfg.get("trim_start_seconds", 0),
        trim_end_seconds=fe_cfg.get("trim_end_seconds", 0),
    )
    if stop_after == "frames":
        print(f"\n✓ Stopped after frame extraction for {video_path.name}\n")
        return

    # ── Step 2: Person Segmentation + Crop Extraction (YOLO-Seg) ────────
    seg_cfg = config.get("segmentation", {})
    print(f"\n{'='*60}")
    print("Step 2/8 - Person Segmentation (YOLO-Seg)")
    print(f"{'='*60}")
    seg_manifest_path = segment_skier(
        frame_dir,
        seg_dir,
        model_name=seg_cfg.get("model", "yolo11x-seg"),
        device=seg_cfg.get("device", "mps"),
        confidence=seg_cfg.get("confidence", 0.5),
        padding_ratio=seg_cfg.get("padding_ratio", 0.15),
        select_strategy=seg_cfg.get("select_strategy", "center"),
    )
    if stop_after == "segmentation":
        print(f"\n✓ Stopped after segmentation for {video_path.name}\n")
        return

    # ── Step 3: Visualization (Pre-Tracking) ────────────────────────────
    viz_cfg = config.get("visualization", {})
    meta = load_frame_meta(frame_dir)
    viz_fps = viz_cfg.get("fps") or meta.get("effective_fps")
    if viz_fps is None:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(str(video_path))
        viz_fps = cap.get(_cv2.CAP_PROP_FPS)
        cap.release()
    if stop_after is None:
        print(f"\n{'='*60}")
        print(f"Step 3/8 - Visualization (Pre-Tracking) ({viz_fps:.2f} fps)")
        print(f"{'='*60}")
        visualize_segmentation_boxes(frame_dir, seg_manifest_path, viz_dir, fps=viz_fps)

    # ── Step 4: Temporal Tracking ───────────────────────────────────────
    trk_cfg = config.get("tracking", {})
    tracking_enabled = trk_cfg.get("enabled", True)
    pose_manifest = seg_manifest_path
    if tracking_enabled:
        print(f"\n{'='*60}")
        print("Step 4/8 - Temporal Tracking")
        print(f"{'='*60}")
        track_manifest_path = track_skier(
            seg_manifest_path,
            frame_dir,
            track_dir,
            w_conf=trk_cfg.get("w_conf", 0.3),
            w_center=trk_cfg.get("w_center", 0.5),
            w_length=trk_cfg.get("w_length", 0.2),
            min_track_frames=trk_cfg.get("min_track_frames", 10),
            smooth_window=trk_cfg.get("smooth_window", 5),
            padding_ratio=seg_cfg.get("padding_ratio", 0.15),
            merge_tracks=trk_cfg.get("merge_tracks", True),
            merge_score_threshold=trk_cfg.get("merge_score_threshold", 0.3),
            merge_min_detection_conf=trk_cfg.get("merge_min_detection_conf", 0.0),
            merge_threshold_adaptive=trk_cfg.get("merge_threshold_adaptive", False),
            merge_threshold_low=trk_cfg.get("merge_threshold_low", 0.5),
            merge_threshold_high=trk_cfg.get("merge_threshold_high", 0.6),
            merge_threshold_min_overlap_ratio=trk_cfg.get("merge_threshold_min_overlap_ratio", 0.3),
            of_gap_fill_enabled=trk_cfg.get("of_gap_fill_enabled", True),
            of_min_gap_for_fill=trk_cfg.get("of_min_gap_for_fill", 0),
            of_drift_guard_px=trk_cfg.get("of_drift_guard_px", 0.0),
            of_reanchor_min_conf=trk_cfg.get("of_reanchor_min_conf", 0.0),
            of_trace_filter_enabled=trk_cfg.get("of_trace_filter_enabled", True),
            optical_flow_method=trk_cfg.get("optical_flow_method", "auto"),
            flow_max_extrapolate_frames=trk_cfg.get("flow_max_extrapolate_frames", 30),
            flow_min_keypoints=trk_cfg.get("flow_min_keypoints", 5),
            identity_guard_enabled=trk_cfg.get("identity_guard_enabled", True),
            identity_guard_max_jump_px=trk_cfg.get("identity_guard_max_jump_px", 150.0),
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
            cotracker_enabled=trk_cfg.get("cotracker_enabled", False),
            cotracker_min_gap=trk_cfg.get("cotracker_min_gap", 20),
            cotracker_resize_h=trk_cfg.get("cotracker_resize_h", 320),
            conflict_of_multiplier=trk_cfg.get("conflict_of_multiplier", 3.0),
            conflict_stickiness_multiplier=trk_cfg.get("conflict_stickiness_multiplier", 0.2),
            conflict_velocity_multiplier=trk_cfg.get("conflict_velocity_multiplier", 1.0),
            kalman_reinit_gap=trk_cfg.get("kalman_reinit_gap", 0),
            w_size=trk_cfg.get("w_size", 0.0),
            w_color=trk_cfg.get("w_color", 0.0),
            of_trace_single_cand_reanchor=trk_cfg.get("of_trace_single_cand_reanchor", False),
            of_trace_bidirectional=trk_cfg.get("of_trace_bidirectional", False),
        )
        pose_manifest = track_manifest_path

    if stop_after == "tracking":
        print(f"\n✓ Stopped after tracking for {video_path.name}\n")
        return

    # ── Step 5: 2D Pose Estimation ──────────────────────────────────────
    p2d_cfg = config.get("pose_2d", {})
    print(f"\n{'='*60}")
    print("Step 5/8 - 2D Pose Estimation (YOLO)")
    print(f"{'='*60}")
    poses_2d_path = estimate_2d_poses_yolo(
        frame_dir,
        pose_2d_dir,
        model_name=p2d_cfg.get("yolo_model", "yolo11x-pose"),
        device=p2d_cfg.get("device", "mps"),
        confidence=p2d_cfg.get("bbox_threshold", 0.25),
        kpt_thr=p2d_cfg.get("keypoint_threshold", 0.3),
        imgsz=p2d_cfg.get("imgsz", 1280),
        segmentation_manifest=pose_manifest,
    )

    # ── Step 6: 3D Pose Lifting ─────────────────────────────────────────
    p3d_cfg = config.get("pose_3d", {})
    print(f"\n{'='*60}")
    print("Step 6/8 - 3D Pose Lifting (MotionBERT)")
    print(f"{'='*60}")
    poses_3d_path = lift_to_3d(
        poses_2d_path,
        pose_3d_dir,
        model_name=p3d_cfg.get("model", "motionbert"),
        checkpoint_path=p3d_cfg.get("checkpoint"),
        checkpoint_url=p3d_cfg.get("checkpoint_url"),
        device=p3d_cfg.get("device", "mps"),
        receptive_field=p3d_cfg.get("receptive_field", 243),
        model_kwargs=p3d_cfg.get("model_kwargs"),
        batch_size=p3d_cfg.get("batch_size", 16),
    )

    # ── Step 7: 2D Pose Overlay Visualization ──────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 7/8 - 2D Pose Overlay Visualization ({viz_fps:.2f} fps)")
    print(f"{'='*60}")
    visualize_2d_overlay(
        frame_dir,
        poses_2d_path,
        viz_dir,
        skeleton=True,
        draw_bbox=True,
        fps=viz_fps,
    )

    # ── Step 8: Visualization (Post-Tracking + 3D) ─────────────────────
    viz_cfg = config.get("visualization", {})
    print(f"\n{'='*60}")
    print(f"Step 8/8 - Visualization (Post-Tracking + 3D) ({viz_fps:.2f} fps)")
    print(f"{'='*60}")
    if tracking_enabled:
        visualize_tracking_boxes(frame_dir, track_manifest_path, viz_dir, fps=viz_fps)

    if viz_cfg.get("render_3d", True):
        visualize_3d_side_by_side(frame_dir, poses_3d_path, viz_dir, fps=viz_fps)
        if viz_cfg.get("export_format", "html") == "html":
            visualize_3d_plotly(poses_3d_path, viz_dir, fps=int(viz_fps))

    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete for {video_path.name}")
    print(f"  Outputs in: {base_out}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="3D Pose Estimation Pipeline for Freeride Skiing"
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to a video file (required with --stage; omit for batch processing)",
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--stage", "-S",
        choices=list(STAGES.keys()),
        help="Run a single pipeline stage independently (requires explicit video path)",
    )
    parser.add_argument(
        "--max-videos", "-n", type=int, default=None, help="Max number of videos to process (batch mode only)"
    )
    parser.add_argument(
        "--stop-after",
        choices=["frames", "segmentation", "tracking"],
        default=None,
        help="Stop the pipeline after this stage (skips pose estimation and visualization)",
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Quick test mode: 5 fps, 50 frames max"
    )
    args = parser.parse_args()

    # Validate required arguments for --stage mode
    if args.stage and not args.video:
        parser.error("--stage requires an explicit video path (positional argument)")

    config = load_config(args.config)

    # Override config for quick testing
    if args.test:
        if args.stage == "frames" or not args.stage:
            print("🚀 Quick test mode enabled: fps=5, max_frames=50")
            config.setdefault("frame_extraction", {})
            config["frame_extraction"]["fps"] = 5
            config["frame_extraction"]["max_frames"] = 50

    # ── Single Stage Mode ──────────────────────────────────────────────
    if args.stage:
        run_single_stage(args.stage, args.video, config)

    # ── Full Pipeline or Batch Mode ────────────────────────────────────
    else:
        if args.video:
            # Single video, full pipeline (or partial if --stop-after)
            run_pipeline(args.video, config, stop_after=args.stop_after)
        else:
            # Batch: process all videos in raw_videos/
            raw_dir = Path(config.get("raw_videos_dir", "raw_videos"))
            videos = sorted(raw_dir.glob("*.mp4"))
            if not videos:
                print(f"No .mp4 files found in {raw_dir}")
                sys.exit(1)

            if args.max_videos:
                videos = videos[: args.max_videos]

            stage_label = f"up to {args.stop_after}" if args.stop_after else "full pipeline"
            print(f"Processing {len(videos)} videos from {raw_dir} ({stage_label})\n")
            for video in videos:
                run_pipeline(video, config, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
