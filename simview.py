"""Render tracked player/ball positions as a moving-dot view.

If a homography is supplied, renders a true fixed bird's-eye pitch diagram (positions
transformed from camera pixels to real-world meters). Otherwise falls back to a
camera-frame-relative view (dots plotted directly in original pixel space).
"""
import numpy as np
import cv2
import imageio

from calibration import (PITCH_LENGTH, PITCH_WIDTH, CENTER_CIRCLE_R,
                          PENALTY_AREA_LEN, PENALTY_AREA_WIDTH, GOAL_AREA_LEN, GOAL_AREA_WIDTH,
                          PENALTY_SPOT_DIST, GOAL_WIDTH, GOAL_DEPTH,
                          apply_homography)

PITCH_GREEN = (40, 110, 40)
LINE_COLOR = (255, 255, 255)
BALL_COLOR = (255, 255, 255)
TEAM_COLORS = [(60, 90, 235), (235, 150, 40)]  # BGR, must match detect.py's TEAM_COLORS

PX_PER_M = 10
MARGIN = 40  # leaves room for the goal depth (2m) to render fully outside the pitch boundary
CANVAS_W = int(PITCH_LENGTH * PX_PER_M) + 2 * MARGIN
CANVAS_H = int(PITCH_WIDTH * PX_PER_M) + 2 * MARGIN


def _world_to_px(x, y):
    px = int((x + PITCH_LENGTH / 2) * PX_PER_M) + MARGIN
    py = int((y + PITCH_WIDTH / 2) * PX_PER_M) + MARGIN
    return px, py


def px_to_world(px, py):
    x = (px - MARGIN) / PX_PER_M - PITCH_LENGTH / 2
    y = (py - MARGIN) / PX_PER_M - PITCH_WIDTH / 2
    return x, y


def _draw_pitch(canvas):
    L, W = PITCH_LENGTH, PITCH_WIDTH

    def line(x1, y1, x2, y2):
        cv2.line(canvas, _world_to_px(x1, y1), _world_to_px(x2, y2), LINE_COLOR, 2)

    line(-L / 2, -W / 2, L / 2, -W / 2)
    line(L / 2, -W / 2, L / 2, W / 2)
    line(L / 2, W / 2, -L / 2, W / 2)
    line(-L / 2, W / 2, -L / 2, -W / 2)
    line(0, -W / 2, 0, W / 2)

    cx, cy = _world_to_px(0, 0)
    cv2.circle(canvas, (cx, cy), int(CENTER_CIRCLE_R * PX_PER_M), LINE_COLOR, 2)
    cv2.circle(canvas, (cx, cy), 3, LINE_COLOR, -1)

    for side in (1, -1):
        x_edge = side * L / 2
        x_pen = x_edge - side * PENALTY_AREA_LEN
        line(x_edge, -PENALTY_AREA_WIDTH / 2, x_pen, -PENALTY_AREA_WIDTH / 2)
        line(x_pen, -PENALTY_AREA_WIDTH / 2, x_pen, PENALTY_AREA_WIDTH / 2)
        line(x_pen, PENALTY_AREA_WIDTH / 2, x_edge, PENALTY_AREA_WIDTH / 2)
        x_goal = x_edge - side * GOAL_AREA_LEN
        line(x_edge, -GOAL_AREA_WIDTH / 2, x_goal, -GOAL_AREA_WIDTH / 2)
        line(x_goal, -GOAL_AREA_WIDTH / 2, x_goal, GOAL_AREA_WIDTH / 2)
        line(x_goal, GOAL_AREA_WIDTH / 2, x_edge, GOAL_AREA_WIDTH / 2)

        # Penalty arc ("the D"): the part of a circle around the penalty spot that falls
        # outside the box — sample the full circle and keep only points beyond the box edge.
        px_spot = x_edge - side * PENALTY_SPOT_DIST
        arc_pts = []
        for deg in range(0, 360, 4):
            rad = np.radians(deg)
            x = px_spot + CENTER_CIRCLE_R * np.cos(rad)
            y = CENTER_CIRCLE_R * np.sin(rad)
            if (side == 1 and x <= x_pen) or (side == -1 and x >= x_pen):
                arc_pts.append((x, y))
        for i in range(len(arc_pts) - 1):
            line(*arc_pts[i], *arc_pts[i + 1])

        # Goal: small open rectangle sticking out beyond the goal line.
        x_back = x_edge + side * GOAL_DEPTH
        line(x_edge, -GOAL_WIDTH / 2, x_back, -GOAL_WIDTH / 2)
        line(x_back, -GOAL_WIDTH / 2, x_back, GOAL_WIDTH / 2)
        line(x_back, GOAL_WIDTH / 2, x_edge, GOAL_WIDTH / 2)


def _dot(canvas, x, y, color, r):
    cv2.circle(canvas, (int(x), int(y)), r, color, -1)
    cv2.circle(canvas, (int(x), int(y)), r, (0, 0, 0), 1)


def render_pitch_diagram():
    """A blank top-down field diagram at native canvas scale, for calibration click UI."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), PITCH_GREEN, dtype=np.uint8)
    _draw_pitch(canvas)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def render_sim_video(frame_positions, width, height, fps, output_path, homography=None, progress_cb=None, speed=1.0):
    """speed: playback rate multiplier. 0.5 = half speed (slow motion), 1.0 = real-time."""
    writer = imageio.get_writer(output_path, fps=fps * speed, codec="libx264", pixelformat="yuv420p")
    total = len(frame_positions)

    for i, fp in enumerate(frame_positions):
        if homography is not None:
            canvas = np.full((CANVAS_H, CANVAS_W, 3), PITCH_GREEN, dtype=np.uint8)
            _draw_pitch(canvas)

            player_pts = [(x, y) for x, y, _ in fp["players"]]
            teams = [t for _, _, t in fp["players"]]
            world_players = apply_homography(homography, player_pts)
            for (wx, wy), team in zip(world_players, teams):
                _dot(canvas, *_world_to_px(wx, wy), TEAM_COLORS[team], 10)

            if fp["ball"]:
                (wx, wy), = apply_homography(homography, [fp["ball"]])
                _dot(canvas, *_world_to_px(wx, wy), BALL_COLOR, 7)
        else:
            # No calibration: plain background, no pitch markings. Drawing a fake halfway
            # line/circle here would look like a real top-down field when it's actually
            # just raw camera pixels — misleading about which side of the pitch is which.
            canvas = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)

            for x, y, team in fp["players"]:
                _dot(canvas, x, y, TEAM_COLORS[team], 14)
            if fp["ball"]:
                _dot(canvas, fp["ball"][0], fp["ball"][1], BALL_COLOR, 8)

            cv2.putText(canvas, "NOT CALIBRATED - camera-relative positions only", (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        if progress_cb and total:
            progress_cb((i + 1) / total)

    writer.close()
