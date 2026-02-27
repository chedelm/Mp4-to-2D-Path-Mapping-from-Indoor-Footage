import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_path_plot(xy: np.ndarray, out_path: str):
    """
    xy: Nx2 array of path points
    """
    plt.figure()
    plt.plot(xy[:, 0], xy[:, 1])
    plt.title("Estimated Camera Path (Prototype)")
    plt.xlabel("X (arbitrary units)")
    plt.ylabel("Y (arbitrary units)")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def save_path_csv(xy: np.ndarray, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "x", "y"])
        for i, (x, y) in enumerate(xy):
            w.writerow([i, float(x), float(y)])