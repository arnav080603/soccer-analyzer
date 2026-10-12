"""Homography calibration: map camera pixels to real-world pitch meters.

World coordinate convention (100x60yd / 91.44x54.86m high school pitch): origin at center
spot, x = along the pitch length (goal-to-goal), y = along the pitch width (touchline-to-touchline).
"""
import json
import os
import cv2
import numpy as np

# Label goals "Goal A"/"Goal B" and sidelines "Sideline A"/"Sideline B" — pick either end
# as "A", just stay consistent across the points you click for one clip.
LANDMARKS = {
    "Center of the field": (0, 0),
    "Halfway line at Sideline A": (0, -27.43),
    "Halfway line at Sideline B": (0, 27.43),
    "Center circle edge, Sideline A side": (0, -9.15),
    "Center circle edge, Sideline B side": (0, 9.15),
    "Center circle edge, toward Goal A": (-9.15, 0),
    "Center circle edge, toward Goal B": (9.15, 0),
    "Penalty spot, Goal A": (34.72, 0),
    "Penalty spot, Goal B": (-34.72, 0),
    "Corner flag, Goal A / Sideline A": (45.72, -27.43),
    "Corner flag, Goal A / Sideline B": (45.72, 27.43),
    "Corner flag, Goal B / Sideline A": (-45.72, -27.43),
    "Corner flag, Goal B / Sideline B": (-45.72, 27.43),
    "18-yard box corner, Goal A / Sideline A": (29.22, -20.16),
    "18-yard box corner, Goal A / Sideline B": (29.22, 20.16),
    "18-yard box corner, Goal B / Sideline A": (-29.22, -20.16),
    "18-yard box corner, Goal B / Sideline B": (-29.22, 20.16),
}

# High school fields run smaller than the FIFA-regulation 105x68m pitch — 100x60 yards
# is the common NFHS size. Goal width and center-circle radius are fixed by the Laws of
# the Game regardless of overall field size, so only the outer dimensions change here.
PITCH_LENGTH = 91.44  # 100 yards
PITCH_WIDTH = 54.86  # 60 yards
CENTER_CIRCLE_R = 9.15
PENALTY_AREA_LEN = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LEN = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DIST = 11.0
GOAL_WIDTH = 7.32
GOAL_DEPTH = 2.0  # visual only, not a regulation dimension


def calib_path(clip_name):
    base = os.path.splitext(clip_name)[0]
    return os.path.join(os.path.dirname(__file__), "data", f"{base}_calib.json")


def save_calibration(clip_name, points):
    """points: list of {"label", "image": (x,y), "world": (X,Y)}"""
    with open(calib_path(clip_name), "w") as f:
        json.dump(points, f, indent=2)


def load_calibration(clip_name):
    path = calib_path(clip_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_homography(points):
    img_pts = np.array([p["image"] for p in points], dtype=np.float32)
    world_pts = np.array([p["world"] for p in points], dtype=np.float32)
    # With exactly 4 points there's no redundancy — a single imprecise click distorts the
    # whole transform. With 5+, use RANSAC so a few extra points can outvote one bad click.
    if len(points) >= 5:
        H, _ = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    else:
        H, _ = cv2.findHomography(img_pts, world_pts)
    return H


def apply_homography(H, points):
    """points: list of (x,y) image pixels -> list of (X,Y) world meters."""
    if not points:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return [(float(p[0][0]), float(p[0][1])) for p in out]
