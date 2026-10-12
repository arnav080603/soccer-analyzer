"""YOLOv8 + ByteTrack player/ball tracking over a soccer clip.

Uses ByteTrack (built into ultralytics) instead of independent per-frame detection: it
associates detections across frames using confidence-tiered matching (high-confidence boxes
matched first, low-confidence ones used to recover occluded tracks) and Kalman-filters each
track's motion. On top of that we gap-fill each track's timeline (so a player stays on
screen through brief missed-detection frames instead of flickering out), apply per-track
exponential smoothing to the plotted position, and classify each track's team from its
average jersey color using a trainable classifier (falls back to k-means until it has
enough corrected examples — see teammodel.py).
"""
import os
import cv2
import numpy as np
import imageio
from collections import defaultdict
from ultralytics import YOLO

from teammodel import TeamModel

_model = None
TRACKER_CFG = os.path.join(os.path.dirname(__file__), "bytetrack_custom.yaml")

def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

# COCO class ids we care about: 0=person, 32=sports ball
PERSON_CLS, BALL_CLS = 0, 32
BALL_COLOR = (255, 255, 255)
TEAM_COLORS = [(60, 90, 235), (235, 150, 40)]  # BGR: team 0 red-ish, team 1 blue-ish
MAX_BALL_GAP = 15  # frames; interpolate ball position across gaps up to this long
SMOOTH_ALPHA = 0.35  # exponential smoothing factor for tracked positions; lower = smoother


def _jersey_color(frame, box):
    """Mean BGR color of a player's torso region, with grass-green pixels masked out."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    ty1, ty2 = int(y1 + h * 0.15), int(y1 + h * 0.55)
    crop = frame[max(0, ty1):max(0, ty2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return (128.0, 128.0, 128.0)
    b, g, r = crop[..., 0].astype(int), crop[..., 1].astype(int), crop[..., 2].astype(int)
    grass = (g > r + 15) & (g > b + 15)
    mask = ~grass
    pixels = crop[mask] if mask.sum() >= 10 else crop.reshape(-1, 3)
    return tuple(pixels.mean(axis=0))


def _classify_teams(avg_colors, clip_name):
    """Team label per track id, from a trained TeamModel if available, else k-means fallback."""
    ids = list(avg_colors.keys())
    if not ids:
        return {}
    model = TeamModel(clip_name)
    if model.is_trained():
        preds = model.predict([avg_colors[i] for i in ids])
        return dict(zip(ids, preds))
    if len(ids) >= 2:
        data = np.array([avg_colors[i] for i in ids], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, _ = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        return dict(zip(ids, labels.flatten().tolist()))
    return {ids[0]: 0}


def _stitch_tracks(person_tracks, max_gap=120, base_dist=35, dist_per_frame=4):
    """Merge track fragments that likely belong to the same player: one track ending near
    where another begins shortly after. ByteTrack assigns a fresh ID whenever it loses a
    player (occlusion, a missed detection, a brief overlap with another player) for longer
    than its buffer, which fragments a single real player into many short-lived tracks.
    This is a standard post-hoc "tracklet linking" step: greedily pair each track's end
    with the nearest still-unclaimed start within a gap/distance budget (allowing more
    drift for longer gaps), unioning them into one canonical id.
    Returns {orig_track_id: canonical_track_id}.
    """
    ids = list(person_tracks.keys())
    starts = {tid: person_tracks[tid][0] for tid in ids}
    ends = {tid: person_tracks[tid][-1] for tid in ids}

    parent = {tid: tid for tid in ids}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    candidates = []
    for a in ids:
        a_end_f, a_end_x, a_end_y = ends[a]
        for b in ids:
            if a == b:
                continue
            b_start_f, b_start_x, b_start_y = starts[b]
            gap = b_start_f - a_end_f
            if 0 < gap <= max_gap:
                dist = ((a_end_x - b_start_x) ** 2 + (a_end_y - b_start_y) ** 2) ** 0.5
                if dist <= base_dist + dist_per_frame * gap:
                    candidates.append((gap, dist, a, b))
    candidates.sort(key=lambda c: (c[0], c[1]))

    claimed_end, claimed_start = set(), set()
    for gap, dist, a, b in candidates:
        if a in claimed_end or b in claimed_start or find(a) == find(b):
            continue
        parent[find(b)] = find(a)
        claimed_end.add(a)
        claimed_start.add(b)

    return {tid: find(tid) for tid in ids}


def _fill_gaps(records):
    """records: list of (frame_idx, x, y) sorted by frame_idx.
    Returns a dense {frame_idx: (x, y)} covering every frame from the track's first to last
    appearance, linearly interpolating across any gaps — so a track stays visible through
    brief missed-detection frames instead of flickering out.
    """
    if not records:
        return {}
    dense = {}
    for i in range(len(records) - 1):
        f0, x0, y0 = records[i]
        f1, x1, y1 = records[i + 1]
        dense[f0] = (x0, y0)
        for f in range(f0 + 1, f1):
            t = (f - f0) / (f1 - f0)
            dense[f] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    last = records[-1]
    dense[last[0]] = (last[1], last[2])
    return dense


def _smooth_track(dense):
    """dense: {frame_idx: (x, y)} -> smoothed {frame_idx: (x, y)}"""
    smoothed = {}
    prev = None
    for frame_idx in sorted(dense):
        x, y = dense[frame_idx]
        if prev is None:
            sx, sy = x, y
        else:
            sx = SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * prev[0]
            sy = SMOOTH_ALPHA * y + (1 - SMOOTH_ALPHA) * prev[1]
        smoothed[frame_idx] = (sx, sy)
        prev = (sx, sy)
    return smoothed


def _interpolate_ball(ball_by_frame):
    result = list(ball_by_frame)
    last_idx = None
    for i, v in enumerate(ball_by_frame):
        if v is not None:
            if last_idx is not None and 1 < i - last_idx <= MAX_BALL_GAP:
                x0, y0 = ball_by_frame[last_idx]
                x1, y1 = v
                for j in range(last_idx + 1, i):
                    t = (j - last_idx) / (i - last_idx)
                    result[j] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            last_idx = i
    return result


def _render_outputs(input_path, output_path, raw_frames, merge_map, smoothed_by_track,
                     active_by_frame, track_team, ball_interp, fps, progress_cb=None):
    """Draw the annotated video and build frame_positions from already-computed tracking
    data. Shared by process_video (first run) and rerender_teams (after a correction) so a
    team-color fix doesn't require re-running YOLO.
    """
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")
    cap = cv2.VideoCapture(input_path)
    person_frames = ball_frames = 0
    frame_positions = []
    total = len(raw_frames)

    for idx, fr in enumerate(raw_frames):
        ok, frame = cap.read()
        if not ok:
            break

        for p in fr["players"]:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), TEAM_COLORS[track_team.get(merge_map[p["track_id"]], 0)], 2)

        players_out = [(*smoothed_by_track[tid][idx], track_team.get(tid, 0)) for tid in active_by_frame[idx]]
        if players_out:
            person_frames += 1

        if fr["ball_box"]:
            bx1, by1, bx2, by2 = fr["ball_box"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), BALL_COLOR, 2)
            ball_frames += 1

        frame_positions.append({"players": players_out, "ball": ball_interp[idx]})

        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if progress_cb and total:
            progress_cb((idx + 1) / total)

    cap.release()
    writer.close()
    return person_frames, ball_frames, frame_positions


def process_video(input_path, output_path, conf=0.1, progress_cb=None):
    """Returns dict with frame_count, fps, person_frames, ball_frames, frame_positions
    (list per frame of {"players": [(x,y,team_id),...], "ball": (x,y) or None}), frame_width/height,
    track_avg_colors and track_thumbnails (for the team-correction UI), and track_team plus
    the raw tracking state (needed by rerender_teams to redraw after a correction without
    re-running YOLO).
    """
    model = get_model()
    model.predictor = None  # drop any tracker state left over from a previous clip

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_frames = []  # per frame: {"players": [{"box":.., "color":.., "track_id":..}], "ball_box": .. or None}
    ball_by_frame = []
    person_tracks = defaultdict(list)  # track_id -> [(frame_idx, x, y), ...]
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = frame_count
        frame_count += 1

        results = model.track(frame, persist=True, conf=conf, classes=[PERSON_CLS, BALL_CLS],
                               tracker=TRACKER_CFG, verbose=False)[0]
        players, ball_box = [], None
        for box in results.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id == PERSON_CLS:
                if score < 0.25:
                    continue
                color = _jersey_color(frame, (x1, y1, x2, y2))
                players.append({"box": (x1, y1, x2, y2), "color": color, "track_id": track_id})
                person_tracks[track_id].append((idx, (x1 + x2) / 2, y2))
            elif cls_id == BALL_CLS:
                ball_box = (x1, y1, x2, y2)

        raw_frames.append({"players": players, "ball_box": ball_box})
        ball_by_frame.append(((ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2) if ball_box else None)
        if progress_cb and total:
            progress_cb(0.4 * frame_count / total)

    cap.release()

    # Stitch fragmented tracks back together before anything else keys off track_id, so
    # box colors, gap-filling, and smoothing all see one continuous id per real player.
    merge_map = _stitch_tracks(person_tracks)
    stitched_tracks = defaultdict(list)
    for tid, recs in person_tracks.items():
        stitched_tracks[merge_map[tid]].extend(recs)
    for tid in stitched_tracks:
        stitched_tracks[tid].sort(key=lambda r: r[0])
    person_tracks = stitched_tracks

    filled_by_track = {tid: _fill_gaps(recs) for tid, recs in person_tracks.items()}
    smoothed_by_track = {tid: _smooth_track(dense) for tid, dense in filled_by_track.items()}
    active_by_frame = defaultdict(list)
    for tid, dense in filled_by_track.items():
        for frame_idx in dense:
            active_by_frame[frame_idx].append(tid)

    track_colors = defaultdict(list)
    track_boxes = defaultdict(dict)  # tid -> {frame_idx: box}, for team-correction thumbnails
    for idx, fr in enumerate(raw_frames):
        for p in fr["players"]:
            tid = merge_map[p["track_id"]]
            track_colors[tid].append(p["color"])
            track_boxes[tid][idx] = p["box"]
    avg_colors = {tid: tuple(np.mean(cs, axis=0)) for tid, cs in track_colors.items()}
    track_team = _classify_teams(avg_colors, os.path.basename(input_path))

    # Grab one representative crop per track (its median detected frame) for the correction UI.
    thumb_frame = {tid: sorted(boxes)[len(boxes) // 2] for tid, boxes in track_boxes.items()}
    thumb_needed = defaultdict(list)
    for tid, f in thumb_frame.items():
        thumb_needed[f].append(tid)
    track_thumbnails = {}
    cap_thumb = cv2.VideoCapture(input_path)
    idx = 0
    while thumb_needed:
        ok, frame = cap_thumb.read()
        if not ok:
            break
        for tid in thumb_needed.pop(idx, []):
            x1, y1, x2, y2 = track_boxes[tid][idx]
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                track_thumbnails[tid] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        idx += 1
    cap_thumb.release()

    ball_interp = _interpolate_ball(ball_by_frame)

    def render_cb(frac):
        if progress_cb:
            progress_cb(0.4 + 0.6 * frac)

    person_frames, ball_frames, frame_positions = _render_outputs(
        input_path, output_path, raw_frames, merge_map, smoothed_by_track,
        active_by_frame, track_team, ball_interp, fps, progress_cb=render_cb)

    return {
        "frame_count": frame_count,
        "fps": fps,
        "person_frames": person_frames,
        "ball_frames": ball_frames,
        "frame_positions": frame_positions,
        "frame_width": w,
        "frame_height": h,
        "track_avg_colors": avg_colors,
        "track_thumbnails": track_thumbnails,
        "track_team": track_team,
        # internal state, reused by rerender_teams:
        "_raw_frames": raw_frames,
        "_merge_map": merge_map,
        "_smoothed_by_track": smoothed_by_track,
        "_active_by_frame": active_by_frame,
        "_ball_interp": ball_interp,
    }


def rerender_teams(input_path, output_path, stats, track_team, progress_cb=None):
    """Redraw the annotated video with an updated track_team mapping (e.g. after a
    correction), reusing the tracking already computed by process_video — no YOLO re-run.
    Returns the updated frame_positions (for regenerating the position sim).
    """
    _, _, frame_positions = _render_outputs(
        input_path, output_path, stats["_raw_frames"], stats["_merge_map"],
        stats["_smoothed_by_track"], stats["_active_by_frame"], track_team,
        stats["_ball_interp"], stats["fps"], progress_cb=progress_cb)
    return frame_positions
