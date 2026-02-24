"""3D visualization of skier pose and ski geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# COCO skeleton connections for drawing
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
    frame_paths = sorted(
        list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.png"))
    )

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
    frame_paths = sorted(
        list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.png"))
    )

    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    # Read first frame to get dimensions
    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]

    out_path = output_dir / "tracking_boxes.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    selected_track_id = manifest.get("selected_track_id", -1)

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
        conf = fdata.get("confidence", 0.0)
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
    fps: float = 30.0,
) -> Path:
    """Draw 2D keypoints and bounding boxes overlaid on frames and write a video."""
    import cv2
    from tqdm import tqdm

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(poses_2d_path) as f:
        data = json.load(f)
    frames_data = data["frames"]

    frame_paths = sorted(
        list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.png"))
    )

    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    # Read first frame to get dimensions
    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]

    out_path = output_dir / "overlay_2d.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for fpath, fdata in tqdm(
        zip(frame_paths, frames_data), total=len(frame_paths), desc="Overlay 2D"
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
                    if len(bbox) >= 5 and bbox[4] > 0:
                        cv2.putText(
                            img, f"{bbox[4]:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        )

        # Draw skeleton
        if skeleton:
            for i, j in SKELETON_CONNECTIONS:
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
    from plotly.subplots import make_subplots

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(poses_3d_path) as f:
        data = json.load(f)

    frames_data = data["frames"]
    keypoint_names = data.get("keypoint_names", [f"joint_{i}" for i in range(17)])

    # Build animation frames
    plotly_frames = []
    for fdata in frames_data:
        kpts = np.array(fdata["keypoints_3d"])  # (17, 3)
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
        for i, j in SKELETON_CONNECTIONS:
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
    kpts0 = np.array(first["keypoints_3d"])

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
