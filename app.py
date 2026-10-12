import json
import os
import cv2
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from detect import process_video, rerender_teams, TEAM_COLORS
from trim import trim_video
from simview import render_sim_video, render_pitch_diagram, px_to_world
from calibration import save_calibration, load_calibration, compute_homography
from teammodel import TeamModel

TEAM_NAMES = ["Red", "Blue"]  # must match TEAM_COLORS order in detect.py

BASE = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE, "clips")
PROCESSED_DIR = os.path.join(BASE, "processed")
DATA_DIR = os.path.join(BASE, "data")

EVENT_TYPES = ["Touch", "Pass", "Shot", "Tackle", "Goal"]
EVENT_COLORS = {"Touch": "#888", "Pass": "#4da3ff", "Shot": "#ff4d4d", "Tackle": "#ffa64d", "Goal": "#33cc66"}

st.set_page_config(page_title="Soccer Clip Analyzer", layout="wide")
st.title("⚽ Soccer Clip Analyzer")


def events_path(clip_name):
    return os.path.join(DATA_DIR, f"{os.path.splitext(clip_name)[0]}.json")


def load_events(clip_name):
    path = events_path(clip_name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_events(clip_name, events):
    with open(events_path(clip_name), "w") as f:
        json.dump(events, f, indent=2)


def get_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps else 0


CPU_DETECT_FPS = 17  # benchmarked YOLOv8n inference speed on this machine's CPU


def estimate_detect_seconds(path):
    cap = cv2.VideoCapture(path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / CPU_DETECT_FPS


# --- Sidebar: clip selection / upload ---
st.sidebar.header("Clips")
uploaded = st.sidebar.file_uploader("Upload a clip", type=["mp4", "mov", "m4v"])
if uploaded is not None:
    dest = os.path.join(CLIPS_DIR, uploaded.name)
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Saved {uploaded.name}")

clips = sorted(f for f in os.listdir(CLIPS_DIR) if f.lower().endswith((".mp4", ".mov", ".m4v")))
if not clips:
    st.info("Upload a clip in the sidebar to get started.")
    st.stop()

selected = st.sidebar.radio("Select clip", clips)
clip_path = os.path.join(CLIPS_DIR, selected)

if "events" not in st.session_state or st.session_state.get("_clip") != selected:
    st.session_state.events = load_events(selected)
    st.session_state._clip = selected

duration = get_duration(clip_path)

col_video, col_tag = st.columns([2, 1])

with col_video:
    st.video(clip_path)

with col_tag:
    st.subheader("Tag an event")
    ts = st.number_input("Timestamp (seconds)", min_value=0.0, max_value=max(duration, 0.0), step=0.5)
    etype = st.selectbox("Event type", EVENT_TYPES)
    if st.button("Add event"):
        st.session_state.events.append({"time": ts, "type": etype})
        st.session_state.events.sort(key=lambda e: e["time"])
        save_events(selected, st.session_state.events)
        st.rerun()

st.subheader("Event timeline")
if st.session_state.events:
    for i, ev in enumerate(st.session_state.events):
        c1, c2, c3 = st.columns([1, 3, 1])
        c1.markdown(f"**{ev['time']:.1f}s**")
        c2.markdown(f":{'blue' if ev['type']=='Pass' else 'red' if ev['type']=='Shot' else 'gray'}[{ev['type']}]")
        if c3.button("Delete", key=f"del_{i}"):
            st.session_state.events.pop(i)
            save_events(selected, st.session_state.events)
            st.rerun()
else:
    st.caption("No events tagged yet.")

st.divider()
st.subheader("Extract a highlight clip")
st.caption("For a full game, trim to just the segment you want before running detection on it — detection on 90+ minutes is impractically slow.")
col_start, col_end = st.columns(2)
trim_start = col_start.number_input("Start (seconds)", min_value=0.0, max_value=max(duration, 0.0), step=1.0, key="trim_start")
trim_end = col_end.number_input("End (seconds)", min_value=0.0, max_value=max(duration, 0.0), value=min(30.0, duration), step=1.0, key="trim_end")

if st.button("Extract clip"):
    if trim_end <= trim_start:
        st.error("End must be after start.")
    else:
        base = os.path.splitext(selected)[0]
        out_name = f"{base}_trim_{int(trim_start)}-{int(trim_end)}s.mp4"
        out_path = os.path.join(CLIPS_DIR, out_name)
        progress = st.progress(0.0)

        def trim_cb(frac):
            progress.progress(min(frac, 1.0))

        with st.spinner(f"Extracting {trim_end - trim_start:.0f}s clip..."):
            trim_video(clip_path, out_path, trim_start, trim_end, progress_cb=trim_cb)
        st.success(f"Saved {out_name} — select it in the sidebar.")
        st.rerun()

st.divider()
st.subheader("Bird's-eye calibration (optional)")
st.caption("For each point: click a spot on the field diagram (left), then click the matching spot in the video (right). "
           "Repeat for 6-8 points spread across the frame — with only the technical minimum of 4, one imprecise click can badly distort the whole transform; "
           "more points let the math outvote a bad click instead of being stuck with it.")

if st.session_state.get("_calib_clip") != selected:
    cap = cv2.VideoCapture(clip_path)
    ok, first_frame = cap.read()
    cap.release()
    st.session_state._calib_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB) if ok else None
    st.session_state._calib_clip = selected
    st.session_state.calib_points = load_calibration(selected) or []
    st.session_state.calib_pending_world = None

frame_rgb = st.session_state._calib_frame
if frame_rgb is not None:
    orig_h, orig_w = frame_rgb.shape[:2]
    disp_w = 440
    vid_scale = orig_w / disp_w
    disp_h = int(orig_h / vid_scale)
    disp_img = Image.fromarray(frame_rgb).resize((disp_w, disp_h))

    diagram = render_pitch_diagram()
    diag_orig_h, diag_orig_w = diagram.shape[:2]
    diag_disp_w = 440
    diag_scale = diag_orig_w / diag_disp_w
    diag_disp_h = int(diag_orig_h / diag_scale)
    diag_img = Image.fromarray(diagram).resize((diag_disp_w, diag_disp_h))

    col_diag, col_vid = st.columns(2)
    with col_diag:
        st.caption("1. Click the field diagram")
        diag_coord = streamlit_image_coordinates(diag_img, key=f"diag_{selected}")
    with col_vid:
        st.caption("2. Click the matching spot in the video")
        vid_coord = streamlit_image_coordinates(disp_img, key=f"vid_{selected}")

    if diag_coord and diag_coord != st.session_state.get("_last_diag_click"):
        st.session_state._last_diag_click = diag_coord
        st.session_state.calib_pending_world = px_to_world(diag_coord["x"] * diag_scale, diag_coord["y"] * diag_scale)
        st.rerun()

    if st.session_state.get("calib_pending_world"):
        wx, wy = st.session_state.calib_pending_world
        st.info(f"Field point set at ({wx:.1f}, {wy:.1f})m — now click the matching spot in the video.")

    if vid_coord and vid_coord != st.session_state.get("_last_vid_click") and st.session_state.get("calib_pending_world"):
        st.session_state._last_vid_click = vid_coord
        img_x, img_y = vid_coord["x"] * vid_scale, vid_coord["y"] * vid_scale
        wx, wy = st.session_state.calib_pending_world
        st.session_state.calib_points.append(
            {"label": f"Point {len(st.session_state.calib_points) + 1}", "image": [img_x, img_y], "world": [wx, wy]})
        st.session_state.calib_pending_world = None
        st.rerun()

    if st.session_state.calib_points:
        for i, p in enumerate(st.session_state.calib_points):
            c1, c2 = st.columns([4, 1])
            c1.caption(f"{p['label']} — image ({p['image'][0]:.0f}, {p['image'][1]:.0f}) → world ({p['world'][0]:.1f}, {p['world'][1]:.1f})")
            if c2.button("Remove", key=f"calib_rm_{i}"):
                st.session_state.calib_points.pop(i)
                st.rerun()

        n = len(st.session_state.calib_points)
        if n >= 4:
            if n < 6:
                st.caption(f"{n} points — works, but add a couple more (6+) so one imprecise click can't distort the whole transform.")
            if st.button("Save calibration"):
                save_calibration(selected, st.session_state.calib_points)
                st.success("Calibration saved — the next detection run will use a true bird's-eye view.")
        else:
            st.caption(f"Need at least 4 points ({n}/4).")

st.divider()
st.subheader("Auto-detection (YOLOv8)")
st.caption("Detects players and the ball frame-by-frame using a pretrained model. Works best on stable, non-panning footage.")

est_seconds = estimate_detect_seconds(clip_path)
if est_seconds > 120:
    st.warning(f"Estimated ~{est_seconds/60:.0f} min to process this clip at its current length. Consider extracting a shorter highlight above first.")
else:
    st.caption(f"Estimated processing time: ~{est_seconds:.0f}s")

_calib_now = load_calibration(selected)
if not _calib_now or len(_calib_now) < 4:
    st.warning("This clip isn't calibrated yet — the position sim below will fall back to a camera-relative view "
               "(raw pixel positions, not real pitch coordinates). Calibrate above first if you want an accurate bird's-eye view.")

det_out = os.path.join(PROCESSED_DIR, f"det_{selected}")
sim_out = os.path.join(PROCESSED_DIR, f"sim_{selected}")

sim_speed = st.select_slider("Position sim playback speed", options=[0.25, 0.5, 1.0], value=0.5,
                              format_func=lambda s: f"{s}x" + (" (slow motion)" if s < 1.0 else " (real-time)"))


def _render_sim(stats, frame_positions):
    calib_points = load_calibration(selected)
    homography = compute_homography(calib_points) if calib_points and len(calib_points) >= 4 else None
    render_sim_video(frame_positions, stats["frame_width"], stats["frame_height"],
                      stats["fps"], sim_out, homography=homography, speed=sim_speed)


if st.button("Run detection on this clip"):
    progress = st.progress(0.0)

    def cb(frac):
        progress.progress(min(frac, 1.0))

    with st.spinner("Running YOLOv8 over the clip..."):
        stats = process_video(clip_path, det_out, progress_cb=cb)

    with st.spinner("Rendering position sim..."):
        _render_sim(stats, stats["frame_positions"])

    st.session_state.last_stats = stats
    st.session_state.last_stats_clip = selected
    st.session_state.track_team = dict(stats["track_team"])
    st.session_state.pending_corrections = {}
    st.rerun()

if st.session_state.get("last_stats_clip") == selected:
    stats = st.session_state.last_stats
    st.success("Detection complete")
    m1, m2, m3 = st.columns(3)
    m1.metric("Frames processed", stats["frame_count"])
    m2.metric("Frames with player(s)", stats["person_frames"])
    m3.metric("Frames with ball detected", stats["ball_frames"])
    st.video(det_out)
    st.video(sim_out)

    st.divider()
    st.subheader("Correct team assignments (trains the model)")
    n_examples = len(TeamModel(selected).examples)
    pending = st.session_state.setdefault("pending_corrections", {})
    st.caption(f"Click the correct team for any player it got wrong — queues the fix. Nothing retrains or redraws until you hit \"Apply corrections\" below, "
               f"so you can flag everything first. Trained on {n_examples} correction(s) so far for this clip — each clip trains its own model, since different games have different jersey colors.")

    cols = st.columns(4)
    for i, (tid, thumb) in enumerate(stats["track_thumbnails"].items()):
        team = st.session_state.track_team.get(tid, 0)
        with cols[i % 4]:
            flagged = tid in pending
            st.image(thumb, caption=f"Currently: {TEAM_NAMES[team]}" + (" (queued)" if flagged else ""), width=100)
            other = 1 - team
            if st.button(f"→ {TEAM_NAMES[other]}", key=f"fix_team_{tid}", disabled=flagged):
                pending[tid] = other
                st.session_state.track_team[tid] = other
                st.rerun()

    if pending:
        st.info(f"{len(pending)} correction(s) queued.")
        c1, c2 = st.columns(2)
        if c1.button("Apply corrections & retrain"):
            examples = [(stats["track_avg_colors"][tid], team) for tid, team in pending.items()]
            redraw_progress = st.progress(0.0)
            with st.spinner("Retraining and redrawing..."):
                TeamModel(selected).add_corrections(examples)
                frame_positions = rerender_teams(clip_path, det_out, stats, st.session_state.track_team,
                                                   progress_cb=lambda f: redraw_progress.progress(min(f, 1.0)))
                _render_sim(stats, frame_positions)
            stats["frame_positions"] = frame_positions
            st.session_state.pending_corrections = {}
            st.rerun()
        if c2.button("Clear queued corrections"):
            for tid in pending:
                st.session_state.track_team[tid] = stats["track_team"][tid]
            st.session_state.pending_corrections = {}
            st.rerun()

elif os.path.exists(det_out):
    st.caption("Previously generated detection output:")
    st.video(det_out)
    if os.path.exists(sim_out):
        st.video(sim_out)
