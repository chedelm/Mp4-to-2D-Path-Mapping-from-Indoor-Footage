import cv2
import numpy as np
from typing import Optional, Tuple

def detect_points(gray: np.ndarray, max_corners: int, quality_level: float, min_distance: int, block_size: int) -> Optional[np.ndarray]:
    """
    Detect trackable points in a grayscale frame.
    Returns Nx1x2 float32 array or None.
    """
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
    )
    return pts


def estimate_translation(prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray,
                         lk_win_size: int, lk_max_level: int) -> Tuple[float, float, np.ndarray]:
    """
    Tracks points using Lucas-Kanade optical flow and estimates median translation dx, dy (in pixels).
    Robust to OpenCV point shapes like (N,1,2) vs (N,2).
    Returns (dx, dy, curr_pts_good) where curr_pts_good is (N,1,2) float32.
    """
    lk_params = dict(
        winSize=(lk_win_size, lk_win_size),
        maxLevel=lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    if curr_pts is None or status is None:
        return 0.0, 0.0, prev_pts

    status = status.reshape(-1)

    # Select good points
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]

    # Force shape to (N,2)
    good_prev = good_prev.reshape(-1, 2)
    good_curr = good_curr.reshape(-1, 2)

    if good_prev.shape[0] < 8:
        curr_good = good_curr.reshape(-1, 1, 2).astype(np.float32)
        return 0.0, 0.0, curr_good

    delta = good_curr - good_prev  # (N,2)
    dx = float(np.median(delta[:, 0]))
    dy = float(np.median(delta[:, 1]))

    curr_good = good_curr.reshape(-1, 1, 2).astype(np.float32)
    return dx, dy, curr_good