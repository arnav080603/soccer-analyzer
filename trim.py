"""Extract a time-range sub-clip from a longer video."""
import cv2
import imageio


def trim_video(input_path, output_path, start_sec, end_sec, progress_cb=None):
    """Write frames from start_sec to end_sec into a new H.264 mp4. Returns frame count written."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")

    total_to_write = max(end_frame - start_frame, 0)
    written = 0
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        written += 1
        frame_idx += 1
        if progress_cb and total_to_write:
            progress_cb(written / total_to_write)

    cap.release()
    writer.close()
    return written
