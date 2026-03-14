"""3D visualization of skier pose and ski geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pose_3d.keypoint_converter import H36M_SKELETON_CONNECTIONS, get_hip_indices


# COCO skeleton connections for 2D overlay drawing
SKELETON_CONNECTIONS_2D = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

# 3D views are rendered in MotionBERT/H36M joint convention.
SKELETON_CONNECTIONS_3D = H36M_SKELETON_CONNECTIONS


def visualize_segmentation_boxes(
    frame_dir: str | Path,
    segmentation_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    box_color: tuple[int, int, int] = (0, 255, 0),
    box_thickness: int = 2,
    fps: int = 30,
) -> Path:
    """Reconstruct video with YOLO segmentation bounding boxes overlaid.

    Parameters
    ----------
    frame_dir : directory containing original frames.
    segmentation_manifest_path : path to the segmentation.json manifest.
    output_dir : output directory for the video.
    box_color : BGR color tuple for bounding box (default: green).
    box_thickness : thickness of bounding box lines.
    fps : frames per second for output video.

    Returns
    -------
    Path to the output video.
    """
    import cv2
    from tqdm import tqdm

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation manifest
    with open(segmentation_manifest_path) as f:
        manifest = json.load(f)
    frames_data = manifest["frames"]

    # Get frame paths
    frame_paths = sorted(list(frame_dir.glob("frame_*.jpg")) + list(frame_dir.glob("frame_*.png")))

    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    # Read first frame to get dimensions
    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]

    out_path = output_dir / "segmentation_boxes.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for fpath, fdata in tqdm(
        zip(frame_paths, frames_data), total=len(frame_paths), desc="Segmentation Boxes"
    ):
        img = cv2.imread(str(fpath))

        if fdata.get("detected", False):
            # Draw bounding box
            x1, y1, x2, y2 = fdata["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)

            # Draw padded bounding box with dashed effect (different color)
            px1, py1, px2, py2 = fdata["bbox_padded"]
            # For dashed line, draw multiple small segments
            for i in range(0, max(px2 - px1, py2 - py1), 10):
                if i + 5 < px2 - px1:
                    cv2.line(img, (px1 + i, py1), (px1 + i + 5, py1), (100, 149, 237), 1)  # cornflower blue
                if i + 5 < py2 - py1:
                    cv2.line(img, (px2, py1 + i), (px2, py1 + i + 5), (100, 149, 237), 1)
                if i + 5 < px2 - px1:
                    cv2.line(img, (px1 + i, py2), (px1 + i + 5, py2), (100, 149, 237), 1)
                if i + 5 < py2 - py1:
                    cv2.line(img, (px1, py1 + i), (px1, py1 + i + 5), (100, 149, 237), 1)

            # Add confidence text
            conf = fdata.get("confidence", 0.0)
            text = f"Conf: {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 6), (x1 + text_size[0] + 4, y1), box_color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(img)

    writer.release()
    print(f"  → Segmentation boxes video saved to {out_path}")
    return out_path


def visualize_tracking_boxes(
    frame_dir: str | Path,
    tracking_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    box_color: tuple[int, int, int] = (255, 165, 0),  # orange for tracked
    interpolated_color: tuple[int, int, int] = (255, 0, 255),  # magenta for interpolated
    box_thickness: int = 2,
    fps: int = 30,
) -> Path:
    """Reconstruct video with tracked/smoothed bounding boxes overlaid.

    Parameters
    ----------
    frame_dir : directory containing original frames.
    tracking_manifest_path : path to the tracking.json manifest.
    output_dir : output directory for the video.
    box_color : BGR color tuple for detected bounding boxes (default: orange).
    interpolated_color : BGR color for interpolated boxes (default: magenta).
    box_thickness : thickness of bounding box lines.
    fps : frames per second for output video.

    Returns
    -------
    Path to the output video.
    """
    import cv2
    from tqdm import tqdm

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tracking manifest
    with open(tracking_manifest_path) as f:
        manifest = json.load(f)
    frames_data = manifest["frames"]

    # Get frame paths
    frame_paths = sorted(list(frame_dir.glob("frame_*.jpg")) + list(frame_dir.glob("frame_*.png")))

    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    # Read first frame to get dimensions
    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]

    out_path = output_dir / "tracking_boxes.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for fpath, fdata in tqdm(
        zip(frame_paths, frames_data), total=len(frame_paths), desc="Tracking Boxes"
    ):
        img = cv2.imread(str(fpath))

        # Determine box color based on interpolation
        is_interpolated = fdata.get("interpolated", False)
        color = interpolated_color if is_interpolated else box_color

        # Draw bounding box
        x1, y1, x2, y2 = fdata["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        # Draw padded bounding box with dashed effect
        px1, py1, px2, py2 = fdata["bbox_padded"]
        dash_color = tuple(int(c * 0.7) for c in color)  # darker shade
        for i in range(0, max(px2 - px1, py2 - py1), 10):
            if i + 5 < px2 - px1:
                cv2.line(img, (px1 + i, py1), (px1 + i + 5, py1), dash_color, 1)
            if i + 5 < py2 - py1:
                cv2.line(img, (px2, py1 + i), (px2, py1 + i + 5), dash_color, 1)
            if i + 5 < px2 - px1:
                cv2.line(img, (px1 + i, py2), (px1 + i + 5, py2), dash_color, 1)
            if i + 5 < py2 - py1:
                cv2.line(img, (px1, py1 + i), (px1, py1 + i + 5), dash_color, 1)

        # Add track info text
        track_id = fdata.get("track_id", -1)
        status = "Interp" if is_interpolated else "Detect"
        text = f"Track:{track_id} {status}"

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 6), (x1 + text_size[0] + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(img)

    writer.release()
    print(f"  → Tracking boxes video saved to {out_path}")
    return out_path


def visualize_2d_overlay(
    frame_dir: str | Path,
    poses_2d_path: str | Path,
    output_dir: str | Path,
    *,
    skeleton: bool = True,
    draw_bbox: bool = True,
    tracking_manifest_path: str | Path | None = None,
    fps: float = 30.0,
) -> Path:
    """Draw 2D keypoints and bounding boxes overlaid on frames and write a video.

    Parameters
    ----------
    frame_dir : directory containing original frames.
    poses_2d_path : path to the poses_2d.json manifest.
    output_dir : output directory for the video.
    skeleton : whether to draw skeleton connections.
    draw_bbox : whether to draw bounding boxes.
    tracking_manifest_path : optional path to tracking.json to draw buffered bounding boxes.
    fps : frames per second for output video.

    Returns
    -------
    Path to the output video.
    """
    import cv2
    from tqdm import tqdm

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(poses_2d_path) as f:
        data = json.load(f)
    frames_data = data["frames"]

    # Load tracking manifest if provided
    tracking_frames_data = None
    if tracking_manifest_path:
        tracking_path = Path(tracking_manifest_path)
        if tracking_path.exists():
            with open(tracking_path) as f:
                tracking_data = json.load(f)
            tracking_frames_data = tracking_data.get("frames", [])

    frame_paths = sorted(list(frame_dir.glob("frame_*.jpg")) + list(frame_dir.glob("frame_*.png")))

    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    # Read first frame to get dimensions
    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]

    out_path = output_dir / "overlay_2d.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for idx, (fpath, fdata) in enumerate(tqdm(
        zip(frame_paths, frames_data), total=len(frame_paths), desc="Overlay 2D"
    )):
        img = cv2.imread(str(fpath))
        kpts = np.array(fdata["keypoints"])  # (17, 3)

        # Draw buffered bounding box from tracking if available
        if tracking_frames_data and idx < len(tracking_frames_data):
            track_fdata = tracking_frames_data[idx]
            bbox_padded = track_fdata.get("bbox_padded", [])
            if len(bbox_padded) >= 4:
                px1, py1, px2, py2 = int(bbox_padded[0]), int(bbox_padded[1]), int(bbox_padded[2]), int(bbox_padded[3])
                if px2 > px1 and py2 > py1:
                    # Draw padded bounding box with dashed effect in blue
                    dash_color = (100, 149, 237)  # cornflower blue
                    for i in range(0, max(px2 - px1, py2 - py1), 10):
                        if i + 5 < px2 - px1:
                            cv2.line(img, (px1 + i, py1), (px1 + i + 5, py1), dash_color, 1)
                        if i + 5 < py2 - py1:
                            cv2.line(img, (px2, py1 + i), (px2, py1 + i + 5), dash_color, 1)
                        if i + 5 < px2 - px1:
                            cv2.line(img, (px1 + i, py2), (px1 + i + 5, py2), dash_color, 1)
                        if i + 5 < py2 - py1:
                            cv2.line(img, (px1, py1 + i), (px1, py1 + i + 5), dash_color, 1)

        # Draw skeleton
        if skeleton:
            for i, j in SKELETON_CONNECTIONS_2D:
                if kpts[i, 2] > 0.3 and kpts[j, 2] > 0.3:
                    pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                    pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                    cv2.line(img, pt1, pt2, (255, 128, 0), 2)

        # Draw keypoints
        for k in range(kpts.shape[0]):
            if kpts[k, 2] > 0.3:
                cx, cy = int(kpts[k, 0]), int(kpts[k, 1])
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        writer.write(img)

    writer.release()
    print(f"  → 2D overlay video saved to {out_path}")
    return out_path


def visualize_3d_plotly(
    poses_3d_path: str | Path,
    output_dir: str | Path,
    *,
    fps: int = 30,
) -> Path:
    """Create an interactive 3D visualization using Plotly."""
    import plotly.graph_objects as go

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(poses_3d_path) as f:
        data = json.load(f)

    frames_data = data["frames"]
    keypoint_names = data.get("keypoint_names", [f"joint_{i}" for i in range(17)])
    skeleton_connections = _choose_3d_skeleton(keypoint_names)

    # Build animation frames
    plotly_frames = []
    for fdata in frames_data:
        kpts = np.array(fdata["keypoints_3d"], dtype=np.float32)  # (17, 3)
        kpts[:, 1] *= -1.0
        x, y, z = kpts[:, 0], kpts[:, 1], kpts[:, 2]

        # Joints as scatter
        joint_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            marker=dict(size=5, color="red"),
            text=keypoint_names,
            textposition="top center",
            textfont=dict(size=8),
            name="joints",
        )

        # Skeleton lines
        skeleton_x, skeleton_y, skeleton_z = [], [], []
        for i, j in skeleton_connections:
            skeleton_x += [x[i], x[j], None]
            skeleton_y += [y[i], y[j], None]
            skeleton_z += [z[i], z[j], None]

        bone_trace = go.Scatter3d(
            x=skeleton_x, y=skeleton_y, z=skeleton_z,
            mode="lines",
            line=dict(color="cyan", width=4),
            name="skeleton",
        )

        plotly_frames.append(go.Frame(data=[joint_trace, bone_trace], name=str(fdata["frame_id"])))

    # Initial state (first frame)
    first = frames_data[0]
    kpts0 = np.array(first["keypoints_3d"], dtype=np.float32)
    kpts0[:, 1] *= -1.0

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=kpts0[:, 0], y=kpts0[:, 1], z=kpts0[:, 2],
                mode="markers",
                marker=dict(size=5, color="red"),
            ),
        ],
        frames=plotly_frames,
        layout=go.Layout(
            title="3D Skier Pose",
            scene=dict(
                xaxis=dict(range=[-1.5, 1.5]),
                yaxis=dict(range=[-1.5, 1.5]),
                zaxis=dict(range=[-1.5, 1.5]),
                aspectmode="cube",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 1000 // fps}}]),
                        dict(label="⏸ Pause", method="animate", args=[[None], {"mode": "immediate"}]),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(args=[[str(f["frame_id"])], {"mode": "immediate"}], label=str(f["frame_id"]), method="animate")
                        for f in frames_data
                    ],
                    currentvalue=dict(prefix="Frame: "),
                )
            ],
        ),
    )

    out_path = output_dir / "pose_3d.html"
    fig.write_html(str(out_path))
    print(f"  → 3D interactive visualization saved to {out_path}")
    return out_path


def visualize_3d_side_by_side(
    frame_dir: str | Path,
    poses_3d_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 30.0,
    source_scale: float = 1.0,
    poses_2d_h36m_path: str | Path | None = None,
    include_h36m_2d_panel: bool = True,
) -> Path:
    """Render source images with 3D (and optional 2D H36M) side panels to MP4."""
    import cv2
    from tqdm import tqdm

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    with open(poses_3d_path) as f:
        poses_data = json.load(f)
    frames_data = poses_data["frames"]
    keypoint_names = poses_data.get("keypoint_names", [f"joint_{i}" for i in range(17)])
    skeleton_connections = _choose_3d_skeleton(keypoint_names)

    frames_2d_h36m = None
    keypoint_names_2d_h36m: list[str] = []
    if include_h36m_2d_panel and poses_2d_h36m_path is not None:
        p2d_h36m_path = Path(poses_2d_h36m_path)
        if p2d_h36m_path.exists():
            with open(p2d_h36m_path) as f:
                h36m_data = json.load(f)
            frames_2d_h36m = h36m_data.get("frames", [])
            keypoint_names_2d_h36m = h36m_data.get("keypoint_names", [])

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise RuntimeError(f"Unable to read first frame: {frame_paths[0]}")
    source_scale = max(1.0, float(source_scale))
    if source_scale > 1.0:
        sample = cv2.resize(
            sample,
            (
                max(1, int(round(sample.shape[1] * source_scale))),
                max(1, int(round(sample.shape[0] * source_scale))),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
    frame_h, frame_w = sample.shape[:2]
    panel_h, panel_w = frame_h, frame_w

    out_path = output_dir / "side_by_side_3d.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    panel_count = 3 if frames_2d_h36m is not None else 2
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w + panel_w * (panel_count - 1), frame_h))

    all_kpts = np.array([f["keypoints_3d"] for f in frames_data], dtype=np.float32)
    axis_extent = float(np.max(np.abs(all_kpts))) + 1e-6

    frame_count = min(len(frame_paths), len(frames_data))
    if frames_2d_h36m is not None:
        frame_count = min(frame_count, len(frames_2d_h36m))
    for idx in tqdm(range(frame_count), total=frame_count, desc="3D Side-by-Side"):
        frame = cv2.imread(str(frame_paths[idx]))
        if frame is None:
            continue
        if source_scale > 1.0:
            frame = cv2.resize(
                frame,
                (
                    max(1, int(round(frame.shape[1] * source_scale))),
                    max(1, int(round(frame.shape[0] * source_scale))),
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        if frame.shape[0] != frame_h or frame.shape[1] != frame_w:
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        kpts = np.array(frames_data[idx]["keypoints_3d"], dtype=np.float32)
        panel = _render_projected_3d_panel(
            kpts,
            keypoint_names=keypoint_names,
            skeleton_connections=skeleton_connections,
            width=panel_w,
            height=panel_h,
            axis_extent=axis_extent,
        )

        if frames_2d_h36m is not None:
            kpts_2d_h36m = np.array(frames_2d_h36m[idx]["keypoints"], dtype=np.float32)
            panel_2d = _render_h36m_2d_panel(
                keypoints_2d=kpts_2d_h36m,
                keypoint_names=keypoint_names_2d_h36m,
                skeleton_connections=H36M_SKELETON_CONNECTIONS,
                width=panel_w,
                height=panel_h,
            )
            composed = cv2.hconcat([frame, panel, panel_2d])
        else:
            composed = cv2.hconcat([frame, panel])
        writer.write(composed)

    writer.release()
    print(f"  3D side-by-side video saved to {out_path}")
    return out_path


def _render_projected_3d_panel(
    keypoints_3d: np.ndarray,
    *,
    keypoint_names: list[str],
    skeleton_connections: list[tuple[int, int]],
    width: int,
    height: int,
    axis_extent: float,
) -> np.ndarray:
    """Project a 3D skeleton to a 2D canvas using a fixed camera view."""
    import cv2

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    margin = int(min(width, height) * 0.1)

    yaw = np.deg2rad(32.0)
    pitch = np.deg2rad(-20.0)
    ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    rot = rx @ ry

    pts = keypoints_3d.astype(np.float32)
    pts[:, 1] *= -1.0
    try:
        left_hip_idx, right_hip_idx = get_hip_indices(keypoint_names)
    except RuntimeError:
        left_hip_idx, right_hip_idx = 11, 12
    center = np.mean(pts[[left_hip_idx, right_hip_idx]], axis=0, keepdims=True)
    pts = pts - center
    pts_rot = (rot @ pts.T).T

    scale = (min(width, height) - 2 * margin) / (2.2 * axis_extent)
    origin_x = width // 2
    origin_y = height // 2

    def to_px(point: np.ndarray) -> tuple[int, int]:
        x = int(origin_x + point[0] * scale)
        y = int(origin_y - point[1] * scale)
        return x, y

    axis_len = axis_extent * 0.9
    axes = [
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([axis_len, 0.0, 0.0], dtype=np.float32), (40, 80, 240)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, axis_len, 0.0], dtype=np.float32), (50, 180, 60)),
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, axis_len], dtype=np.float32), (230, 120, 40)),
    ]
    for start, end, color in axes:
        s = to_px((rot @ start.T).T)
        e = to_px((rot @ end.T).T)
        cv2.line(canvas, s, e, color, 2)

    for i, j in skeleton_connections:
        p1 = to_px(pts_rot[i])
        p2 = to_px(pts_rot[j])
        cv2.line(canvas, p1, p2, (34, 139, 230), 3)

    for point in pts_rot:
        px = to_px(point)
        cv2.circle(canvas, px, 4, (20, 20, 220), -1)

    cv2.putText(canvas, "3D Skeleton", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2)
    return canvas


def _choose_3d_skeleton(keypoint_names: list[str]) -> list[tuple[int, int]]:
    """Select skeleton connectivity for the provided keypoint naming convention."""
    lowered = [name.lower() for name in keypoint_names]
    if "pelvis" in lowered and "thorax" in lowered:
        return SKELETON_CONNECTIONS_3D
    return SKELETON_CONNECTIONS_2D


def _render_h36m_2d_panel(
    keypoints_2d: np.ndarray,
    *,
    keypoint_names: list[str],
    skeleton_connections: list[tuple[int, int]],
    width: int,
    height: int,
) -> np.ndarray:
    """Render a normalized 2D H36M skeleton panel."""
    import cv2

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    pts = keypoints_2d.astype(np.float32)
    if pts.shape != (17, 3):
        cv2.putText(canvas, "2D H36M (invalid)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
        return canvas

    conf_thr = 0.15
    valid = pts[:, 2] > conf_thr
    if np.count_nonzero(valid) < 2:
        cv2.putText(canvas, "2D H36M (no keypoints)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
        return canvas

    xy = pts[:, :2].copy()
    valid_xy = xy[valid]
    center = np.mean(valid_xy, axis=0, keepdims=True)
    xy = xy - center

    extent = np.max(np.abs(valid_xy - center)) + 1e-6
    scale = (min(width, height) * 0.38) / extent

    origin_x = width // 2
    origin_y = height // 2

    def to_px(point_xy: np.ndarray) -> tuple[int, int]:
        x = int(origin_x + point_xy[0] * scale)
        y = int(origin_y + point_xy[1] * scale)
        return x, y

    for i, j in skeleton_connections:
        if pts[i, 2] > conf_thr and pts[j, 2] > conf_thr:
            p1 = to_px(xy[i])
            p2 = to_px(xy[j])
            cv2.line(canvas, p1, p2, (34, 139, 230), 3)

    for k in range(pts.shape[0]):
        if pts[k, 2] > conf_thr:
            p = to_px(xy[k])
            cv2.circle(canvas, p, 4, (20, 20, 220), -1)

    label = "2D H36M"
    if keypoint_names and len(keypoint_names) == 17:
        label = "2D H36M Skeleton"
    cv2.putText(canvas, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2)
    return canvas
