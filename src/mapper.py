import numpy as np
import cv2

def init_canvas(size: int) -> np.ndarray:
    """
    Canvas for accumulated edges (grayscale).
    """
    return np.zeros((size, size), dtype=np.uint8)

def safe_paste_max(canvas: np.ndarray, patch: np.ndarray, top: int, left: int):
    """
    Paste patch onto canvas using max() blending, with boundary checks.
    """
    h, w = patch.shape[:2]
    H, W = canvas.shape[:2]

    y0 = max(0, top)
    x0 = max(0, left)
    y1 = min(H, top + h)
    x1 = min(W, left + w)

    if y0 >= y1 or x0 >= x1:
        return

    py0 = y0 - top
    px0 = x0 - left
    py1 = py0 + (y1 - y0)
    px1 = px0 + (x1 - x0)

    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], patch[py0:py1, px0:px1])

def edges_from_gray(gray: np.ndarray, t1: int, t2: int) -> np.ndarray:
    edges = cv2.Canny(gray, t1, t2)
    return edges

def build_edge_map_step(canvas: np.ndarray, edges: np.ndarray, cx: int, cy: int):
    """
    Places edge image centered at (cx, cy) on canvas (rough “map” accumulation).
    """
    h, w = edges.shape[:2]
    top = cy - h // 2
    left = cx - w // 2
    safe_paste_max(canvas, edges, top, left)