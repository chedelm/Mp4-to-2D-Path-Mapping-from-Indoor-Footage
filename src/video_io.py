import cv2
from typing import Generator, Tuple

def iter_frames(video_path: str, frame_step: int = 1, max_frames: int = 10_000) -> Generator[Tuple[int, any], None, None]:
    """
    Yields (frame_index, frame_bgr).
    frame_step=2 means every 2nd frame is processed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    i = 0
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % frame_step == 0:
            yield i, frame
            yielded += 1
            if yielded >= max_frames:
                break

        i += 1

    cap.release()


def to_gray(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)