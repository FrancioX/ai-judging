"""Background optical flow utilities shared across pipeline stages.

Dark-feature masking (rocks/trees) gives reliable camera-pan estimates
because snow/sky are bright and low-texture. Validated at ~28.5px
venue accuracy in camflow experiments.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def dark_feature_mask(
    gray: np.ndarray,
    bbox: list[float],
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
) -> np.ndarray:
    """Build a mask of dark, textured pixels (rocks/trees) excluding athlete bbox.

    Snow and sky are bright and low-texture → unreliable flow.
    Rocks and trees are dark and high-texture → reliable flow.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale frame.
    bbox : list[float]
        Athlete bounding box [x1, y1, x2, y2] to exclude.
    dark_threshold : int
        Max brightness (0-255) to be considered "dark feature". Default 80.
    min_gradient : float
        Min gradient magnitude to require texture (filters smooth dark areas).
    """
    h, w = gray.shape[:2]

    dark_mask = gray < dark_threshold

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    texture_mask = grad_mag > min_gradient

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    athlete_mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        athlete_mask[y1:y2, x1:x2] = True

    return dark_mask & texture_mask & ~athlete_mask


def background_flow(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    bbox: list[float],
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
    min_dark_pixels: int = 50,
    fallback_percentile: int = 50,
) -> tuple[float, float]:
    """Compute background flow from dark, textured pixels (rocks/trees).

    Uses Farneback dense optical flow. Dark+textured pixels (rocks/trees)
    give accurate camera-pan vectors. Falls back to all-background median
    if fewer than *min_dark_pixels* dark pixels are available.

    Returns
    -------
    (dx, dy)
        Camera pan vector in pixels for this frame pair.
    """
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev,
        gray_curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    h, w = gray_prev.shape[:2]

    dark_mask = dark_feature_mask(
        gray_prev,
        bbox,
        dark_threshold=dark_threshold,
        min_gradient=min_gradient,
    )

    dark_count = int(dark_mask.sum())
    if dark_count >= min_dark_pixels:
        dx = float(np.median(flow[dark_mask, 0]))
        dy = float(np.median(flow[dark_mask, 1]))
        return dx, dy

    # Fallback: all non-athlete background
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    all_bg = np.ones((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        all_bg[y1:y2, x1:x2] = False

    bg_dx = flow[all_bg, 0]
    bg_dy = flow[all_bg, 1]

    if bg_dx.size == 0:
        return 0.0, 0.0

    dx = float(np.percentile(bg_dx, fallback_percentile))
    dy = float(np.percentile(bg_dy, fallback_percentile))
    return dx, dy


def compute_per_frame_background_flow(
    frame_dir: Path,
    frame_files: list[str],
    bboxes: list[list[float]],
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
    min_dark_pixels: int = 50,
    zoom_correct: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
    """Compute per-frame background flow vectors using dark-feature pixels.

    Parameters
    ----------
    frame_dir : Path
        Directory containing the frames.
    frame_files : list[str]
        Ordered list of frame filenames (relative to frame_dir).
    bboxes : list[list[float]]
        Per-frame athlete bounding boxes [x1, y1, x2, y2].
    zoom_correct : bool
        Normalise each frame's flow by sqrt(bbox_area / ref_area) to
        compensate for camera zoom.

    Returns
    -------
    (per_frame_bg, n_dark_frames)
        per_frame_bg: shape (N, 2), flow vector per frame pair (index i = flow
        between frame i-1 and frame i; index 0 is always [0, 0]).
        n_dark_frames: number of frames where dark pixels were used.
    """
    n = len(frame_files)
    per_frame_bg = np.zeros((n, 2), dtype=float)
    n_dark_used = 0

    bbox_areas = np.zeros(n, dtype=float)
    for i, bb in enumerate(bboxes):
        if len(bb) >= 4:
            bw = max(1.0, bb[2] - bb[0])
            bh = max(1.0, bb[3] - bb[1])
            bbox_areas[i] = bw * bh

    ref_area = bbox_areas[bbox_areas > 0].mean() if np.any(bbox_areas > 0) else 1.0

    prev_gray = None
    for i, fname in enumerate(frame_files):
        frame_gray = cv2.imread(str(frame_dir / fname), cv2.IMREAD_GRAYSCALE)
        if frame_gray is None:
            if verbose:
                print(f"  [flow] Frame {i} unreadable: {fname}")
            prev_gray = None
            continue

        if prev_gray is not None:
            dm = dark_feature_mask(
                prev_gray,
                bboxes[i],
                dark_threshold=dark_threshold,
                min_gradient=min_gradient,
            )
            if int(dm.sum()) >= min_dark_pixels:
                n_dark_used += 1

            dx, dy = background_flow(
                prev_gray,
                frame_gray,
                bboxes[i],
                dark_threshold=dark_threshold,
                min_gradient=min_gradient,
                min_dark_pixels=min_dark_pixels,
            )

            if zoom_correct and bbox_areas[i] > 0:
                zoom_factor = float(np.sqrt(bbox_areas[i] / ref_area))
                dx /= zoom_factor
                dy /= zoom_factor

            per_frame_bg[i] = [dx, dy]

        prev_gray = frame_gray

        if verbose and i % 100 == 0:
            print(f"  [flow] {i}/{n} frames processed...", end="\r")

    if verbose:
        print(f"  [flow] Done. Dark pixels used in {n_dark_used}/{n} frames.")

    return per_frame_bg, n_dark_used


def compute_cumulative_background_flow(
    frame_dir: Path,
    frame_files: list[str],
    bboxes: list[list[float]],
    *,
    dark_threshold: int = 80,
    min_gradient: float = 5.0,
    min_dark_pixels: int = 50,
    zoom_correct: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
    """Compute cumulative background flow from frame 0.

    Wrapper around :func:`compute_per_frame_background_flow` that accumulates
    the per-frame vectors into a cumulative displacement curve.

    Returns
    -------
    (cum_flow, n_dark_frames)
        cum_flow: shape (N, 2), cumulative displacement from frame 0.
        n_dark_frames: frames where dark pixels were used.
    """
    per_frame_bg, n_dark_used = compute_per_frame_background_flow(
        frame_dir,
        frame_files,
        bboxes,
        dark_threshold=dark_threshold,
        min_gradient=min_gradient,
        min_dark_pixels=min_dark_pixels,
        zoom_correct=zoom_correct,
        verbose=verbose,
    )
    cum_flow = np.cumsum(per_frame_bg, axis=0)
    return cum_flow, n_dark_used
