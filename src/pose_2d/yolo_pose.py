"""2D pose estimation using YOLO-Pose (Ultralytics).

Single-model detection + 17 COCO keypoints.  Runs natively on MPS
(Apple Silicon), avoiding the mmcv CPU bottleneck.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# COCO 17 body keypoint names (same order as YOLO-Pose output)
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]


def estimate_2d_poses(
    frame_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "yolo11x-pose",
    device: str = "mps",
    confidence: float = 0.25,
    kpt_thr: float = 0.3,
    imgsz: int = 1280,
    batch_size: int = 32,
    segmentation_manifest: str | Path | None = None,
) -> Path:
    """Run YOLO-Pose on all frames: detection + 17 keypoints in one pass.

    If segmentation_manifest is provided, runs on cropped images and remaps
    keypoints back to full-frame coordinates using bbox_padded offsets.

    Parameters
    ----------
    frame_dir : directory containing extracted frames (frame_*.jpg/png) or crops.
    output_dir : where to write poses_2d.json.
    model_name : Ultralytics YOLO-Pose model name (auto-downloads).
    device : inference device — 'mps', 'cuda:0', or 'cpu'.
    confidence : detection confidence threshold.
    kpt_thr : keypoint confidence threshold (for selection, all saved).
    imgsz : inference image size — larger values detect smaller persons
        at the cost of speed/memory.  Use 1280 or 1920 for distant subjects.
    segmentation_manifest : path to segmentation.json for coordinate remapping.
        When provided, reads from crops/ subdirectory and remaps crop-space
        keypoints to full-frame coordinates.

    Returns
    -------
    Path to poses_2d.json with the same schema as rtmpose output.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "⚠  ultralytics not installed — writing a stub result for testing.\n"
            "   Install with:  uv add ultralytics"
        )
        frame_paths = _collect_frames(Path(frame_dir), segmentation_manifest)
        return _write_stub_poses(frame_paths, Path(output_dir), segmentation_manifest)

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation manifest for coordinate remapping
    seg_offsets: dict[int, tuple[int, int]] = {}  # frame_id → (offset_x, offset_y)
    seg_frame_files: dict[int, str] = {}  # frame_id → original frame filename
    seg_crop_files: dict[int, str] = {}  # frame_id → crop filename
    if segmentation_manifest is not None:
        seg_manifest_path = Path(segmentation_manifest)
        with open(seg_manifest_path) as f:
            seg_data = json.load(f)
        for fr in seg_data.get("frames", []):
            bp = fr.get("bbox_padded", [0, 0, 0, 0])
            seg_offsets[fr["frame_id"]] = (bp[0], bp[1])
            seg_frame_files[fr["frame_id"]] = fr["frame_file"]
            seg_crop_files[fr["frame_id"]] = fr["crop_file"]
        # Override frame_dir to read from crops subdirectory
        frame_dir = seg_manifest_path.parent / "crops"

    # Build frame paths list - use manifest order if available
    if segmentation_manifest is not None and seg_crop_files:
        # Use crop files from manifest in the correct order
        frame_paths = [frame_dir / seg_crop_files[i] for i in sorted(seg_crop_files.keys())]
    else:
        # Fall back to globbing directory
        frame_paths = _collect_frames(frame_dir, segmentation_manifest)

    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    print(f"Running YOLO-Pose on {len(frame_paths)} frames …")
    print(f"  Model: {model_name}  |  Device: {device}  |  ImgSz: {imgsz}  |  Conf: {confidence}")

    # Append .pt if needed
    model_file = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
    model = YOLO(model_file)

    results_list: list[dict] = []

    # Process in batches to avoid GPU memory overflow
    pbar = tqdm(total=len(frame_paths), desc="2D Pose (YOLO)")
    for batch_start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[batch_start : batch_start + batch_size]
        batch_strs = [str(fp) for fp in batch_paths]

        stream = model(
            batch_strs,
            device=device,
            conf=confidence,
            imgsz=imgsz,
            verbose=False,
            stream=True,
        )

        for local_idx, result in enumerate(stream):
            idx = batch_start + local_idx
            fpath = frame_paths[idx]

            if result.keypoints is not None and len(result.keypoints) > 0:
                # Take highest-confidence person (first detection after NMS)
                kpts_xy = result.keypoints.xy[0].cpu().numpy()       # (17, 2)
                kpts_conf = result.keypoints.conf[0].cpu().numpy()   # (17,)
                keypoints_with_score = [
                    [float(kpts_xy[i, 0]), float(kpts_xy[i, 1]), float(kpts_conf[i])]
                    for i in range(len(kpts_xy))
                ]

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    bbox = boxes.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                    bbox_score = float(boxes.conf[0].cpu().numpy())
                else:
                    bbox = [0, 0, 0, 0]
                    bbox_score = 0.0
            else:
                keypoints_with_score = [[0.0, 0.0, 0.0]] * 17
                bbox = [0, 0, 0, 0]
                bbox_score = 0.0

            # Remap coordinates from crop space → original frame space
            if idx in seg_offsets:
                ox, oy = seg_offsets[idx]
                keypoints_with_score = [
                    [kp[0] + ox, kp[1] + oy, kp[2]]
                    for kp in keypoints_with_score
                ]
                if isinstance(bbox, list) and len(bbox) >= 4:
                    bbox = [bbox[0] + ox, bbox[1] + oy, bbox[2] + ox, bbox[3] + oy]

            # Use original frame filename if available
            frame_file = seg_frame_files.get(idx, fpath.name)

            results_list.append({
                "frame_id": idx,
                "frame_file": frame_file,
                "keypoints": keypoints_with_score,
                "bbox": bbox + [bbox_score],
            })
            pbar.update(1)

    pbar.close()

    out_path = output_dir / "poses_2d.json"
    with open(out_path, "w") as f:
        json.dump({"keypoint_names": COCO_KEYPOINTS, "frames": results_list}, f, indent=2)

    print(f"  → 2D poses saved to {out_path}")
    return out_path


def render_pose_video(
    frame_dir: str | Path,
    poses_2d_path: str | Path,
    output_path: str | Path,
    *,
    fps: float = 30.0,
    kpt_thr: float = 0.3,
    skeleton: bool = True,
    draw_bbox: bool = True,
) -> Path:
    """Render a video with bounding boxes and 2D pose skeleton overlaid.

    Parameters
    ----------
    frame_dir : directory with original frames.
    poses_2d_path : path to poses_2d.json.
    output_path : path for the output .mp4 video.
    fps : output video frame rate.
    kpt_thr : minimum confidence to draw a keypoint / bone.
    skeleton : whether to draw skeleton connections.
    draw_bbox : whether to draw the person bounding box.

    Returns
    -------
    Path to the output video.
    """
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(poses_2d_path) as f:
        data = json.load(f)
    frames_data = data["frames"]

    frame_paths = _collect_frames(frame_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for fpath, fdata in tqdm(
        zip(frame_paths, frames_data), total=len(frame_paths), desc="Pose Video"
    ):
        img = cv2.imread(str(fpath))
        kpts = np.array(fdata["keypoints"])  # (17, 3)

        # Draw bounding box
        if draw_bbox:
            bbox = fdata.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if len(bbox) >= 5:
                        conf_text = f"{bbox[4]:.2f}"
                        cv2.putText(
                            img, conf_text, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        )

        # Draw skeleton
        if skeleton:
            for i, j in SKELETON_CONNECTIONS:
                if kpts[i, 2] > kpt_thr and kpts[j, 2] > kpt_thr:
                    pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                    pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                    cv2.line(img, pt1, pt2, (255, 128, 0), 2)

        # Draw keypoints
        for k in range(kpts.shape[0]):
            if kpts[k, 2] > kpt_thr:
                cx, cy = int(kpts[k, 0]), int(kpts[k, 1])
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        writer.write(img)

    writer.release()
    print(f"  → Pose video saved to {output_path}")
    return output_path


# ── Helpers ──────────────────────────────────────────────────────────────


def _collect_frames(frame_dir: Path, segmentation_manifest: str | Path | None = None) -> list[Path]:
    """Collect and sort frame image paths.

    If segmentation_manifest is provided, collects crop_*.jpg/png files.
    Otherwise collects frame_*.jpg/png files.
    """
    if segmentation_manifest is not None:
        return sorted(
            list(frame_dir.glob("crop_*.jpg"))
            + list(frame_dir.glob("crop_*.png"))
        )
    return sorted(
        list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.png"))
    )


def _write_stub_poses(
    frame_paths: list[Path],
    output_dir: Path,
    segmentation_manifest: str | Path | None = None,
) -> Path:
    """Write a placeholder JSON when ultralytics is not available."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original frame filenames if using crops
    seg_frame_files: dict[int, str] = {}
    if segmentation_manifest is not None:
        with open(segmentation_manifest) as f:
            seg_data = json.load(f)
        for fr in seg_data.get("frames", []):
            seg_frame_files[fr["frame_id"]] = fr["frame_file"]

    stub = {
        "keypoint_names": COCO_KEYPOINTS,
        "frames": [
            {
                "frame_id": i,
                "frame_file": seg_frame_files.get(i, p.name),
                "keypoints": [[0.0, 0.0, 0.0]] * 17,
                "bbox": [0, 0, 0, 0, 0],
            }
            for i, p in enumerate(frame_paths)
        ],
    }
    out_path = output_dir / "poses_2d.json"
    with open(out_path, "w") as f:
        json.dump(stub, f, indent=2)
    return out_path
