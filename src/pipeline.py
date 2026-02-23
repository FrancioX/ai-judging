"""Main pipeline: video → segmentation (YOLO-Seg) → 2D pose on crops (YOLO-Pose or RTMPose) → visualization.

Quality-first 4-stage pipeline:
1. Frame extraction
2. Person segmentation + crop extraction (YOLO-Seg)
3. 2D pose estimation from crops with coordinate remapping
4. 2D overlay visualization

Disabled stages (re-enable when models are ready):
- 3D pose lifting (MotionBERT)
- Ski detection (GroundingDINO+SAM2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.utils.video import extract_frames, reconstruct_video_with_bboxes
from src.segmentation.yolo_seg import segment_skier
from src.pose_2d.yolo_pose import (
    estimate_2d_poses as estimate_2d_poses_yolo,
    render_pose_video,
)
from src.pose_2d.rtmpose import estimate_2d_poses as estimate_2d_poses_rtmpose
# Disabled imports (re-enable when models are ready):
# from src.pose_3d.lifter import lift_to_3d
# from src.ski_detection.detector import detect_skis
from src.visualization.render import visualize_2d_overlay  # visualize_3d_plotly disabled


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
    viz_dir = base_out / "visualizations" / video_stem

    # ── Step 1: Extract frames ──────────────────────────────────────────
    fe_cfg = config.get("frame_extraction", {})
    print(f"\n{'='*60}")
    print(f"Step 1/4 — Frame Extraction: {video_path.name}")
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

    # ── Step 2: Person Segmentation + Crop Extraction (YOLO-Seg) ────────
    seg_cfg = config.get("segmentation", {})
    print(f"\n{'='*60}")
    print("Step 2/4 — Person Segmentation (YOLO-Seg)")
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

    # ── Step 3: 2D Pose Estimation from Crops ───────────────────────────
    # Pose model runs on high-resolution crops and remaps keypoints back to
    # full-frame coordinates using bbox_padded offsets from Step 2.
    p2d_cfg = config.get("pose_2d", {})
    pose_backend = p2d_cfg.get("backend", "yolo")
    print(f"\n{'='*60}")
    print(f"Step 3/4 — 2D Pose Estimation ({pose_backend})")
    print(f"{'='*60}")
    if pose_backend == "rtmpose":
        poses_2d_path = estimate_2d_poses_rtmpose(
            frame_dir,
            pose_2d_dir,
            model_name=p2d_cfg.get("model", "rtmpose-l_8xb256-420e_coco-256x192"),
            det_model=p2d_cfg.get("det_model", "rtmdet-m"),
            device=p2d_cfg.get("device", "cpu"),
            bbox_thr=p2d_cfg.get("bbox_threshold", 0.5),
            kpt_thr=p2d_cfg.get("keypoint_threshold", 0.3),
            segmentation_manifest=seg_manifest_path,
        )
    else:
        poses_2d_path = estimate_2d_poses_yolo(
            frame_dir,
            pose_2d_dir,
            model_name=p2d_cfg.get("yolo_model", "yolo11x-pose"),
            device=p2d_cfg.get("device", "mps"),
            confidence=p2d_cfg.get("bbox_threshold", 0.25),
            kpt_thr=p2d_cfg.get("keypoint_threshold", 0.3),
            imgsz=p2d_cfg.get("imgsz", 1280),
            segmentation_manifest=seg_manifest_path,
        )

    # ── DISABLED: 3D Pose Lifting ────────────────────────────────────────
    # Re-enable when MotionBERT is fully integrated.
    # p3d_cfg = config.get("pose_3d", {})
    # print(f"\n{'='*60}")
    # print(f"Step X/4 — 3D Pose Lifting")
    # print(f"{'='*60}")
    # poses_3d_path = lift_to_3d(
    #     poses_2d_path,
    #     pose_3d_dir,
    #     model_name=p3d_cfg.get("model", "motionbert"),
    #     device=p3d_cfg.get("device", "mps"),
    #     receptive_field=p3d_cfg.get("receptive_field", 243),
    # )

    # ── DISABLED: Ski Detection ──────────────────────────────────────────
    # Re-enable when GroundingDINO+SAM2 is fully integrated.
    # ski_cfg = config.get("ski_detection", {})
    # print(f"\n{'='*60}")
    # print(f"Step X/4 — Ski Detection")
    # print(f"{'='*60}")
    # ski_path = detect_skis(
    #     frame_dir,
    #     ski_dir,
    #     method=ski_cfg.get("method", "color_segmentation"),
    #     prompt=ski_cfg.get("prompt", "ski"),
    #     device=ski_cfg.get("device", "mps"),
    #     confidence=ski_cfg.get("confidence", 0.35),
    # )

    # ── Step 4: Visualization ────────────────────────────────────────────
    viz_cfg = config.get("visualization", {})
    target_fps = fe_cfg.get("fps")
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video_path))
    original_fps = cap.get(_cv2.CAP_PROP_FPS)
    cap.release()
    viz_fps = viz_cfg.get("fps")
    if viz_fps is None:
        viz_fps = target_fps if target_fps else original_fps
    print(f"\n{'='*60}")
    print("Step 4/4 — Visualization")
    print(f"{'='*60}")

    if viz_cfg.get("overlay_2d", True):
        visualize_2d_overlay(frame_dir, poses_2d_path, viz_dir, fps=viz_fps)

    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete for {video_path.name}")
    print(f"  Outputs in: {base_out}")
    print(f"{'='*60}\n")


def run_pose_only_pipeline(video_path: str | Path, config: dict | None = None) -> None:
    """Extract frames, segment skier, run 2D pose on crops, and output a video with bbox + skeleton.

    Fast pipeline for validating detection and pose quality.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    if config is None:
        config = load_config()

    video_stem = video_path.stem
    base_out = Path(config.get("output_dir", "output"))
    frame_dir = base_out / "frames" / video_stem
    seg_dir = base_out / "segmentation" / video_stem
    pose_2d_dir = base_out / "poses_2d" / video_stem
    viz_dir = base_out / "visualizations" / video_stem

    # ── Step 1: Extract frames ──────────────────────────────────────────
    fe_cfg = config.get("frame_extraction", {})
    target_fps = fe_cfg.get("fps")
    print(f"\n{'='*60}")
    print(f"Step 1/4 — Frame Extraction: {video_path.name}")
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

    # ── Step 2: Person Segmentation + Crop Extraction (YOLO-Seg) ────────
    seg_cfg = config.get("segmentation", {})
    print(f"\n{'='*60}")
    print("Step 2/4 — Person Segmentation (YOLO-Seg)")
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

    # ── Step 3: 2D Pose (YOLO-Pose on crops) ────────────────────────────
    p2d_cfg = config.get("pose_2d", {})
    print(f"\n{'='*60}")
    print("Step 3/4 — 2D Pose Estimation (YOLO-Pose)")
    print(f"{'='*60}")
    poses_2d_path = estimate_2d_poses_yolo(
        frame_dir,
        pose_2d_dir,
        model_name=p2d_cfg.get("yolo_model", "yolo11x-pose"),
        device=p2d_cfg.get("device", "mps"),
        confidence=p2d_cfg.get("bbox_threshold", 0.25),
        kpt_thr=p2d_cfg.get("keypoint_threshold", 0.3),
        imgsz=p2d_cfg.get("imgsz", 1280),
        segmentation_manifest=seg_manifest_path,
    )

    # ── Step 4: Render pose video ───────────────────────────────────────
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video_path))
    original_fps = cap.get(_cv2.CAP_PROP_FPS)
    cap.release()
    video_fps = target_fps if target_fps else original_fps

    print(f"\n{'='*60}")
    print("Step 4/4 — Rendering pose video")
    print(f"{'='*60}")
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_video = viz_dir / f"{video_stem}_pose_2d.mp4"
    render_pose_video(
        frame_dir,
        poses_2d_path,
        out_video,
        fps=video_fps,
    )

    print(f"\n{'='*60}")
    print(f"✓ Pose pipeline complete for {video_path.name}")
    print(f"  Video: {out_video}")
    print(f"  Poses: {poses_2d_path}")
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
    print("Step 2/2 — Person Segmentation (YOLO)")
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
    print("Reconstructing video with bounding boxes")
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


def run_visualize_only_pipeline(video_path: str | Path, config: dict | None = None) -> None:
    """Regenerate visualization outputs from existing frames and pose data."""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    if config is None:
        config = load_config()

    video_stem = video_path.stem
    base_out = Path(config.get("output_dir", "output"))
    frame_dir = base_out / "frames" / video_stem
    pose_2d_dir = base_out / "poses_2d" / video_stem
    viz_dir = base_out / "visualizations" / video_stem

    poses_2d_path = pose_2d_dir / "poses_2d.json"
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frames not found in {frame_dir}")
    if not poses_2d_path.exists():
        raise FileNotFoundError(f"2D poses not found: {poses_2d_path}")

    viz_cfg = config.get("visualization", {})
    fe_cfg = config.get("frame_extraction", {})
    target_fps = fe_cfg.get("fps")
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video_path))
    original_fps = cap.get(_cv2.CAP_PROP_FPS)
    cap.release()
    viz_fps = viz_cfg.get("fps")
    if viz_fps is None:
        viz_fps = target_fps if target_fps else original_fps

    print(f"\n{'='*60}")
    print("Visualization only — regenerating overlay")
    print(f"{'='*60}")

    if viz_cfg.get("overlay_2d", True):
        visualize_2d_overlay(frame_dir, poses_2d_path, viz_dir, fps=viz_fps)

    print(f"\n{'='*60}")
    print(f"✓ Visualization regenerated for {video_path.name}")
    print(f"  Video directory: {viz_dir}")
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
    parser.add_argument(
        "--pose-only", "-p", action="store_true",
        help="Run frame extraction + 2D pose (YOLO-Pose) and render bbox+skeleton video"
    )
    parser.add_argument(
        "--visualize-only", "-v", action="store_true",
        help="Regenerate visualization video from existing frames + poses"
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
    if args.visualize_only:
        pipeline_func = run_visualize_only_pipeline
    elif args.pose_only:
        pipeline_func = run_pose_only_pipeline
    elif args.segment_only:
        pipeline_func = run_segment_only_pipeline
    else:
        pipeline_func = run_pipeline

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

        mode_name = "pose only" if args.pose_only else "segmentation" if args.segment_only else "full pipeline"
        print(f"Processing {len(videos)} videos from {raw_dir} ({mode_name})\n")
        for video in videos:
            pipeline_func(video, config)


if __name__ == "__main__":
    main()
