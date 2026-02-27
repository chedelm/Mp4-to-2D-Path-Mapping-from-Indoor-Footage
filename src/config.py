from dataclasses import dataclass

@dataclass
class Config:
    # Video sampling
    frame_step: int = 2           # process every Nth frame (increase for speed)
    max_frames: int = 2000        # safety cap

    # Optical flow parameters
    max_corners: int = 300
    quality_level: float = 0.01
    min_distance: int = 8
    block_size: int = 7

    lk_win_size: int = 21
    lk_max_level: int = 3

    # Edge detection
    canny_t1: int = 60
    canny_t2: int = 160

    # Map canvas
    canvas_size: int = 900        # output "map" image size (square)
    pixels_per_step: float = 2.5  # scaling of motion → pixels on canvas