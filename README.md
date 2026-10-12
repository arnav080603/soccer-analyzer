# ⚽ Soccer Clip Analyzer

A computer-vision tool that turns ordinary sideline game footage (for example, a Hudl recording) into **tracked players, team assignments and a top-down tactical view**.

## How it works
1. **Detect and track.** [YOLOv8](https://github.com/ultralytics/ultralytics) detects players and the ball in every frame, and **ByteTrack** links those detections into persistent tracks (Kalman-filtered motion, with low-confidence boxes used to recover players hidden behind others).
2. **Clean up.** Gaps in each track are filled so players don't flicker out, and positions are smoothed per track.
3. **Teams.** Each player is classified into a team by jersey color. It starts with k-means clustering, then switches to a trainable classifier that learns from my corrections (`teammodel.py`).
4. **Pitch calibration.** I click four known field points, and a **homography** maps camera pixels to real field coordinates (`calibration.py`).
5. **Tactical view.** Players are re-rendered on a 2D pitch diagram, and the result can be exported as a "simulation" video (`simview.py`).
6. **Event tagging.** I mark touches, passes, shots, tackles and goals on the timeline to build a match log.

All of this sits behind a Streamlit web UI (`app.py`), which can also trim long game videos into short clips (`trim.py`).

## Run it
```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/streamlit run app.py
```
Put game videos in a `clips/` folder (not included, for the players' privacy). The YOLOv8n weights download automatically on first run.

## Why I built it
I play soccer and wanted to see our team's shape and movement the way pro analysts do, using just the video our school already records.

Python · OpenCV · Ultralytics YOLOv8 · ByteTrack · NumPy · Streamlit. Built with help from AI coding tools (Claude Code).
