"""2D pose estimation using RTMPose (via MMPose).

RTMPose is the state-of-the-art real-time pose estimator.
It outputs COCO-WholeBody 133 keypoints (body + hands + face + feet)
or COCO 17-keypoint body poses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def build_detector(model_name: str = "rtmdet-m", device: str = "mps") -> Any:
    """Build an MMDet person detector.

    Requires mmdet to be installed.  Returns an inferencer object.
    """
    try:
        from mmdet.apis import DetInferencer
    except ImportError as exc:
        raise ImportError(
            "mmdet is required for person detection. "
            "Install with:  uv add mmdet"
        ) from exc

    det_configs = {
        "rtmdet-m": "rtmdet_m_8xb32-300e_coco",
    }
    config = det_configs.get(model_name, model_name)
    return DetInferencer(model=config, device=device)


def build_pose_estimator(
    model_name: str = "rtmpose-x", device: str = "mps"
) -> Any:
    """Build an MMPose RTMPose estimator.

    Returns an inferencer object.
    """
    try:
        from mmpose.apis import PoseInferencer
    except ImportError as exc:
        raise ImportError(
            "mmpose is required for 2D pose estimation. "
            "Install with:  uv add mmpose"
        ) from exc

    pose_configs = {
        "rtmpose-t": "rtmpose-t_8xb256-420e_coco-256x192",
        "rtmpose-s": "rtmpose-s_8xb256-420e_coco-256x192",
        "rtmpose-m": "rtmpose-m_8xb256-420e_coco-256x192",
        "rtmpose-l": "rtmpose-l_8xb256-420e_coco-256x192",
        "rtmpose-x": "rtmpose-x_8xb256-700e_coco-384x288",
    }
    config = pose_configs.get(model_name, model_name)
    return PoseInferencer(pose2d=config, device=device)


# COCO 17 body keypoint names for reference
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
    model_name: str = "rtmpose-x",
    det_model: str = "rtmdet-m",
    device: str = "mps",
    bbox_thr: float = 0.5,
    kpt_thr: float = 0.3,
    segmentation_manifest: str | Path | None = None,
) -> Path:
    """Run 2D pose estimation on all frames in a directory.

    If segmentation_manifest is provided, the input frames are assumed to be
    cropped YOLO outputs. The keypoints are remapped back to the original
    full-frame coordinate space using the bbox_padded offsets.

    Saves a JSON file with per-frame keypoints:
    {
      "frames": [
        {
          "frame_id": 0,
          "keypoints": [[x, y, score], ...],   # (17, 3) in original frame coords
          "bbox": [x1, y1, x2, y2, score]
        },
        ...
      ]
    }

    Returns the path to the output JSON.
    """
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation manifest for coordinate remapping
    seg_offsets: dict[int, tuple[int, int]] = {}  # frame_id → (offset_x, offset_y)
    if segmentation_manifest is not None:
        import json as _json
        with open(segmentation_manifest) as f:
            seg_data = _json.load(f)
        for fr in seg_data.get("frames", []):
            bp = fr.get("bbox_padded", [0, 0, 0, 0])
            seg_offsets[fr["frame_id"]] = (bp[0], bp[1])

    # Collect frame paths — support both original frames and crops
    frame_paths = sorted(
        list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.png"))
        + list(frame_dir.glob("crop_*.jpg"))
        + list(frame_dir.glob("crop_*.png"))
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    print(f"Running 2D pose estimation on {len(frame_paths)} frames …")
    print(f"  Model: {model_name}  |  Detector: {det_model}  |  Device: {device}")

    # Build the inferencer (MMPose handles detection + pose jointly)
    try:
        from mmpose.apis import PoseInferencer
        inferencer = PoseInferencer(
            pose2d=model_name,
            det_model=det_model,
            device=device,
        )
    except ImportError:
        print(
            "⚠  mmpose not installed — writing a stub result for testing.\n"
            "   Install the full stack with:  uv add mmpose mmdet mmcv"
        )
        return _write_stub_poses(frame_paths, output_dir)

    results_list: list[dict] = []
    for idx, fpath in enumerate(tqdm(frame_paths, desc="2D Pose")):
        result_generator = inferencer(
            str(fpath),
            show=False,
            bbox_thr=bbox_thr,
            kpt_thr=kpt_thr,
        )
        result = next(result_generator)

        predictions = result.get("predictions", [[]])[0]
        if predictions:
            pred = predictions[0]  # take highest-confidence person
            kpts = np.array(pred["keypoints"]).tolist()
            scores = np.array(pred["keypoint_scores"]).tolist()
            keypoints_with_score = [
                [kpts[i][0], kpts[i][1], scores[i]]
                for i in range(len(kpts))
            ]
            bbox = pred.get("bbox", [0, 0, 0, 0])
            bbox_score = pred.get("bbox_score", 0.0)
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

        results_list.append(
            {
                "frame_id": idx,
                "frame_file": fpath.name,
                "keypoints": keypoints_with_score,
                "bbox": bbox + [bbox_score] if isinstance(bbox, list) else bbox,
            }
        )

    out_path = output_dir / "poses_2d.json"
    with open(out_path, "w") as f:
        json.dump({"keypoint_names": COCO_KEYPOINTS, "frames": results_list}, f, indent=2)

    print(f"  → 2D poses saved to {out_path}")
    return out_path


def _write_stub_poses(
    frame_paths: list[Path], output_dir: Path
) -> Path:
    """Write a placeholder JSON when mmpose is not available."""
    stub = {
        "keypoint_names": COCO_KEYPOINTS,
        "frames": [
            {
                "frame_id": i,
                "frame_file": p.name,
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
