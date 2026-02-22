"""Main pipeline: video → segmentation → 2D pose → 3D pose → ski detection → visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.utils.video import extract_frames, reconstruct_video_with_bboxes
from src.segmentation.yolo_seg import segment_skier
from src.pose_2d.rtmpose import estimate_2d_poses
from src.pose_3d.lifter import lift_to_3d
from src.ski_detection.detector import detect_skis
from src.visualization.render import visualize_segmentation_boxes, visualize_2d_overlay, visualize_3d_plotly


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load pipeline configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(video_path: str | Path, config: dict | None = None) -> None:
    """Execute the full 3D pose estimation pipeline on a single video."""
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
    pose_2d_dir = base_out / "poses_2d" / video_stem
    pose_3d_dir = base_out / "poses_3d" / video_stem
    ski_dir = base_out / "ski_masks" / video_stem
    viz_dir = base_out / "visualizations" / video_stem

    # ── Step 1: Extract frames ──────────────────────────────────────────
    fe_cfg = config.get("frame_extraction", {})
    print(f"\n{'='*60}")
    print(f"Step 1/6 — Frame Extraction: {video_path.name}")
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

    # ── Step 2: Person Segmentation (YOLO) ──────────────────────────────
    seg_cfg = config.get("segmentation", {})
    print(f"\n{'='*60}")
    print(f"Step 2/6 — Person Segmentation (YOLO)")
    print(f"{'='*60}")
    seg_manifest_path = segment_skier(
        frame_dir,
        seg_dir,
        model_name=seg_cfg.get("model", "yolo11x-seg"),
        device=seg_cfg.get("device", "mps"),
        confidence=seg_cfg.get("confidence", 0.5),
        padding_ratio=seg_cfg.get("padding_ratio", 0.15),
        select_strategy=seg_cfg.get("select_strategy", "largest"),
    )

    # ── Step 3: 2D Pose Estimation ──────────────────────────────────────
    p2d_cfg = config.get("pose_2d", {})
    print(f"\n{'='*60}")
    print(f"Step 3/6 — 2D Pose Estimation")
    print(f"{'='*60}")
    # Use cropped frames from segmentation if available
    use_seg = p2d_cfg.get("use_segmentation_bbox", True)
    pose_input_dir = seg_dir / "crops" if use_seg else frame_dir
    poses_2d_path = estimate_2d_poses(
        pose_input_dir,
        pose_2d_dir,
        model_name=p2d_cfg.get("model", "rtmpose-x"),
        det_model=p2d_cfg.get("det_model", "rtmdet-m"),
        device=p2d_cfg.get("device", "mps"),
        bbox_thr=p2d_cfg.get("bbox_threshold", 0.5),
        kpt_thr=p2d_cfg.get("keypoint_threshold", 0.3),
        segmentation_manifest=seg_manifest_path if use_seg else None,
    )

    # ── Step 4: 3D Pose Lifting ─────────────────────────────────────────
    p3d_cfg = config.get("pose_3d", {})
    print(f"\n{'='*60}")
    print(f"Step 4/6 — 3D Pose Lifting")
    print(f"{'='*60}")
    poses_3d_path = lift_to_3d(
        poses_2d_path,
        pose_3d_dir,
        model_name=p3d_cfg.get("model", "motionbert"),
        device=p3d_cfg.get("device", "mps"),
        receptive_field=p3d_cfg.get("receptive_field", 243),
    )

    # ── Step 5: Ski Detection ───────────────────────────────────────────
    ski_cfg = config.get("ski_detection", {})
    print(f"\n{'='*60}")
    print(f"Step 5/6 — Ski Detection")
    print(f"{'='*60}")
    ski_path = detect_skis(
        frame_dir,
        ski_dir,
        method=ski_cfg.get("method", "color_segmentation"),
        prompt=ski_cfg.get("prompt", "ski"),
        device=ski_cfg.get("device", "mps"),
        confidence=ski_cfg.get("confidence", 0.35),
    )

    # ── Step 6: Visualization ───────────────────────────────────────────
    viz_cfg = config.get("visualization", {})
    print(f"\n{'='*60}")
    print(f"Step 6/7 — Visualization")
    print(f"{'='*60}")

    if viz_cfg.get("segmentation_boxes", True):
        visualize_segmentation_boxes(
            frame_dir,
            seg_manifest_path,
            viz_dir,
            fps=viz_cfg.get("fps", 30),
        )

    if viz_cfg.get("overlay_2d", True):
        visualize_2d_overlay(frame_dir, poses_2d_path, viz_dir)

    if viz_cfg.get("render_3d", True):
        visualize_3d_plotly(poses_3d_path, viz_dir, fps=viz_cfg.get("fps", 30))

    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete for {video_path.name}")
    print(f"  Outputs in: {base_out}")
    print(f"{'='*60}\n")


def run_segment_only_pipeline(video_path: str | Path, config: dict | None = None) -> None:
    """Execute only segmentation and create a video with bounding boxes.

    This is useful for quickly checking YOLO segmentation results
    without running the full pipeline.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    if config is None:
        config = load_config()

    # Derive output paths
    video_stem = video_path.stem
    base_out = Path(config.get("output_dir", "output"))
    frame_dir = base_out / "frames" / video_stem
    seg_dir = base_out / "segmentation" / video_stem
    viz_dir = base_out / "visualizations" / video_stem

    # ── Step 1: Extract frames ──────────────────────────────────────────
    fe_cfg = config.get("frame_extraction", {})
    target_fps = fe_cfg.get("fps")
    print(f"\n{'='*60}")
    print(f"Step 1/2 — Frame Extraction: {video_path.name}")
    print(f"{'='*60}")
    extract_frames(
        video_path,
        frame_dir,
        fps=target_fps,
        max_frames=fe_cfg.get("max_frames"),
        fmt=fe_cfg.get("format", "jpg"),
        quality=fe_cfg.get("quality", 95),
        trim_start_seconds=fe_cfg.get("trim_start_seconds", 0),
        trim_end_seconds=fe_cfg.get("trim_end_seconds", 0),
    )

    # ── Step 2: Person Segmentation (YOLO) ──────────────────────────────
    seg_cfg = config.get("segmentation", {})
    print(f"\n{'='*60}")
    print(f"Step 2/2 — Person Segmentation (YOLO)")
    print(f"{'='*60}")
    seg_manifest_path = segment_skier(
        frame_dir,
        seg_dir,
        model_name=seg_cfg.get("model", "yolo11x-seg"),
        device=seg_cfg.get("device", "mps"),
        confidence=seg_cfg.get("confidence", 0.5),
        padding_ratio=seg_cfg.get("padding_ratio", 0.15),
        select_strategy=seg_cfg.get("select_strategy", "largest"),
    )

    # ── Video Reconstruction ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Reconstructing video with bounding boxes")
    print(f"{'='*60}")
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_video = viz_dir / f"{video_stem}_segmentation.mp4"

    # Use the original video FPS if we resampled, otherwise use config viz fps
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # If we resampled, use that fps, otherwise use original
    reconstruction_fps = target_fps if target_fps else original_fps

    reconstruct_video_with_bboxes(
        frame_dir,
        seg_manifest_path,
        output_video,
        fps=reconstruction_fps,
    )

    print(f"\n{'='*60}")
    print(f"✓ Segmentation pipeline complete for {video_path.name}")
    print(f"  Video: {output_video}")
    print(f"  Frames: {frame_dir}")
    print(f"  Manifest: {seg_manifest_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="3D Pose Estimation Pipeline for Freeride Skiing"
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to a video file (or omit to process all videos in raw_videos/)",
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--max-videos", "-n", type=int, default=None, help="Max number of videos to process"
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Quick test mode: 5 fps, 50 frames max"
    )
    parser.add_argument(
        "--segment-only", "-s", action="store_true",
        help="Run only YOLO segmentation and create video with bounding boxes"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Override config for quick testing
    if args.test:
        print("🚀 Quick test mode enabled: fps=5, max_frames=50")
        config.setdefault("frame_extraction", {})
        config["frame_extraction"]["fps"] = 5
        config["frame_extraction"]["max_frames"] = 50

    # Choose which pipeline to run
    pipeline_func = run_segment_only_pipeline if args.segment_only else run_pipeline

    if args.video:
        pipeline_func(args.video, config)
    else:
        raw_dir = Path(config.get("raw_videos_dir", "raw_videos"))
        videos = sorted(raw_dir.glob("*.mp4"))
        if not videos:
            print(f"No .mp4 files found in {raw_dir}")
            sys.exit(1)

        if args.max_videos:
            videos = videos[: args.max_videos]

        mode_name = "segmentation" if args.segment_only else "full pipeline"
        print(f"Processing {len(videos)} videos from {raw_dir} ({mode_name})\n")
        for video in videos:
            pipeline_func(video, config)


if __name__ == "__main__":
    main()
