"""Main pipeline: video → segmentation → 2D pose → 3D pose → ski detection → visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.utils.video import extract_frames
from src.segmentation.yolo_seg import segment_skier
from src.pose_2d.rtmpose import estimate_2d_poses
from src.pose_3d.lifter import lift_to_3d
from src.ski_detection.detector import detect_skis
from src.visualization.render import visualize_2d_overlay, visualize_3d_plotly


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
    print(f"Step 6/6 — Visualization")
    print(f"{'='*60}")

    if viz_cfg.get("overlay_2d", True):
        visualize_2d_overlay(frame_dir, poses_2d_path, viz_dir)

    if viz_cfg.get("render_3d", True):
        visualize_3d_plotly(poses_3d_path, viz_dir, fps=viz_cfg.get("fps", 30))

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
        help="Path to a video file (or omit to process all videos in raw_videos/)",
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--max-videos", "-n", type=int, default=None, help="Max number of videos to process"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.video:
        run_pipeline(args.video, config)
    else:
        raw_dir = Path(config.get("raw_videos_dir", "raw_videos"))
        videos = sorted(raw_dir.glob("*.mp4"))
        if not videos:
            print(f"No .mp4 files found in {raw_dir}")
            sys.exit(1)

        if args.max_videos:
            videos = videos[: args.max_videos]

        print(f"Processing {len(videos)} videos from {raw_dir}\n")
        for video in videos:
            run_pipeline(video, config)


if __name__ == "__main__":
    main()
