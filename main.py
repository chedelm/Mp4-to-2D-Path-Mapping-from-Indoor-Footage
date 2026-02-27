import os
import argparse
import numpy as np
import cv2

from src.config import Config
from src.video_io import iter_frames, to_gray
from src.motion import detect_points, estimate_translation
from src.mapper import init_canvas, edges_from_gray, build_edge_map_step
from src.viz import ensure_dir, save_path_plot, save_path_csv

def parse_args():
    p = argparse.ArgumentParser(description="Prototype: Video → 2D Path + Rough Edge Map (Computer Vision)")
    p.add_argument("--video", required=True, help="Path to input video file (e.g., video/test.mp4)")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--frame_step", type=int, default=None, help="Override frame_step from config")
    p.add_argument("--max_frames", type=int, default=None, help="Override max_frames from config")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config()

    if args.frame_step is not None:
        cfg.frame_step = args.frame_step
    if args.max_frames is not None:
        cfg.max_frames = args.max_frames

    ensure_dir(args.out)

    canvas = init_canvas(cfg.canvas_size)
    center = cfg.canvas_size // 2

    # Path in "map coordinates" (arbitrary)
    x, y = 0.0, 0.0
    path = [(x, y)]

    prev_gray = None
    prev_pts = None

    # Canvas cursor (where we paste edges)
    cx, cy = center, center

    for idx, frame in iter_frames(args.video, frame_step=cfg.frame_step, max_frames=cfg.max_frames):
        gray = to_gray(frame)

        if prev_gray is None:
            prev_gray = gray
            prev_pts = detect_points(gray, cfg.max_corners, cfg.quality_level, cfg.min_distance, cfg.block_size)
            if prev_pts is None:
                # fallback: just continue
                prev_pts = None
            # also place first edges
            edges = edges_from_gray(gray, cfg.canny_t1, cfg.canny_t2)
            build_edge_map_step(canvas, edges, cx, cy)
            continue

        # If we lost points, re-detect
        if prev_pts is None or len(prev_pts) < 20:
            prev_pts = detect_points(prev_gray, cfg.max_corners, cfg.quality_level, cfg.min_distance, cfg.block_size)

        if prev_pts is None:
            # still no points, skip motion update
            prev_gray = gray
            continue

        dx, dy, curr_pts = estimate_translation(prev_gray, gray, prev_pts, cfg.lk_win_size, cfg.lk_max_level)

        # Convert pixel translation into "map steps"
        # In a real SLAM pipeline, this would be much more sophisticated.
        x += (-dx) / 50.0
        y += (-dy) / 50.0
        path.append((x, y))

        # Move canvas cursor too (rough)
        cx += int((-dx) * cfg.pixels_per_step / 10.0)
        cy += int((-dy) * cfg.pixels_per_step / 10.0)

        # Clamp cursor inside canvas
        cx = max(0, min(cfg.canvas_size - 1, cx))
        cy = max(0, min(cfg.canvas_size - 1, cy))

        # Build edge map
        edges = edges_from_gray(gray, cfg.canny_t1, cfg.canny_t2)
        build_edge_map_step(canvas, edges, cx, cy)

        # Update for next iteration
        prev_gray = gray
        prev_pts = curr_pts

    path_np = np.array(path, dtype=np.float32)

    # Save path plot and CSV
    save_path_plot(path_np, os.path.join(args.out, "estimated_path.png"))
    save_path_csv(path_np, os.path.join(args.out, "estimated_path.csv"))

    # Save edge map image
    map_path = os.path.join(args.out, "edge_map.png")
    cv2.imwrite(map_path, canvas)

    print("Done.")
    print(f"Saved: {os.path.join(args.out, 'estimated_path.png')}")
    print(f"Saved: {os.path.join(args.out, 'edge_map.png')}")
    print(f"Saved: {os.path.join(args.out, 'estimated_path.csv')}")

if __name__ == "__main__":
    main()