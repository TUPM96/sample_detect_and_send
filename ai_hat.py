#!/usr/bin/env python3
"""
ai_hat.py – Real-time object detection web app using Hailo AI HAT.

Streams annotated live camera video (MJPEG) on the left panel and shows a
real-time list of detected COCO objects on the right panel, all served from a
lightweight Flask web server.

⚠  This server binds to 0.0.0.0 and is intended for use on a trusted local
   network only.  Do not expose it directly to the public internet.

Usage:
    python ai_hat.py [-m MODEL] [-l LABELS] [-s SCORE] [-p PORT] [-c CAMERA]
"""

import argparse
import json
import threading
import time

import cv2
from flask import Flask, Response, render_template_string, stream_with_context
from picamera2 import Picamera2
from picamera2.devices import Hailo, hailo_architecture

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Shared state written by the inference thread, read by HTTP route handlers.
_lock = threading.Lock()
_latest_jpeg: bytes = b""
_current_detections: list = []

# Events that wake streaming threads when new data is available.
_frame_event = threading.Event()
_detection_event = threading.Event()

# ---------------------------------------------------------------------------
# Helper – extract bounding boxes from Hailo postprocess output
# ---------------------------------------------------------------------------

def extract_detections(hailo_output, video_w: int, video_h: int,
                        class_names: list, threshold: float) -> list:
    """Return a list of detection dicts from the HailoRT postprocess output."""
    results = []
    for class_id, detections in enumerate(hailo_output):
        for det in detections:
            score = float(det[4])
            if score >= threshold:
                # Hailo postprocess output order: (y_min, x_min, y_max, x_max), normalised 0–1
                y0, x0, y1, x1 = det[:4]
                results.append({
                    "label": class_names[class_id],
                    "confidence": round(score * 100),
                    "bbox": [
                        int(x0 * video_w), int(y0 * video_h),
                        int(x1 * video_w), int(y1 * video_h),
                    ],
                    "time": time.strftime("%H:%M:%S"),
                })
    return results

# ---------------------------------------------------------------------------
# Inference thread – captures frames, runs Hailo, encodes JPEG
# ---------------------------------------------------------------------------

def inference_loop(model_path: str, labels_path: str,
                   score_thresh: float, camera_num: int) -> None:
    global _latest_jpeg, _current_detections

    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = f.read().splitlines()

    with Hailo(model_path) as hailo:
        model_h, model_w, _ = hailo.get_input_shape()
        video_w, video_h = 1280, 720

        with Picamera2(camera_num) as picam2:
            main_cfg = {"size": (video_w, video_h), "format": "RGB888"}
            lores_cfg = {"size": (model_w, model_h), "format": "RGB888"}
            config = picam2.create_preview_configuration(
                main_cfg,
                lores=lores_cfg,
                controls={"FrameRate": 30},
            )
            picam2.configure(config)
            picam2.start()

            while True:
                # Capture main and lores frames atomically
                request = picam2.capture_request()
                main_frame = request.make_array("main")
                lores_frame = request.make_array("lores")
                request.release()

                # Run inference on the low-resolution frame
                results = hailo.run(lores_frame)
                dets = extract_detections(
                    results, video_w, video_h, class_names, score_thresh
                )

                # Draw bounding boxes on the main frame
                annotated = main_frame.copy()
                for d in dets:
                    x0, y0, x1, y1 = d["bbox"]
                    label = f"{d['label']} {d['confidence']}%"
                    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(
                        annotated, label,
                        (x0, max(y0 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA,
                    )

                # JPEG-encode (convert RGB → BGR for OpenCV)
                _, buf = cv2.imencode(
                    ".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )

                with _lock:
                    _latest_jpeg = buf.tobytes()
                    _current_detections = dets
                _frame_event.set()
                _detection_event.set()

# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------

def _mjpeg_generator():
    while True:
        _frame_event.wait(timeout=1.0)
        _frame_event.clear()
        with _lock:
            frame = _latest_jpeg
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

# ---------------------------------------------------------------------------
# SSE detection-list generator
# ---------------------------------------------------------------------------

def _sse_generator():
    last_dets: list = []
    while True:
        _detection_event.wait(timeout=1.0)
        _detection_event.clear()
        with _lock:
            current = list(_current_detections)
        if current != last_dets:
            yield f"data: {json.dumps(current)}\n\n"
            last_dets = current

# ---------------------------------------------------------------------------
# HTML template (embedded – no separate templates/ folder needed)
# ---------------------------------------------------------------------------

_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AI HAT – Object Detection</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      display: flex; flex-direction: column; height: 100vh;
      font-family: 'Segoe UI', sans-serif; background: #0f0f0f; color: #eee;
    }
    header {
      display: flex; align-items: center; gap: 10px;
      padding: 10px 18px; background: #1a1a2e;
      font-size: 1.1rem; font-weight: bold; border-bottom: 2px solid #4fc3f7;
    }
    header span { color: #4fc3f7; }
    main { display: flex; flex: 1; overflow: hidden; gap: 8px; padding: 8px; }

    /* ── Video panel ────────────────────────────────── */
    #video-panel {
      flex: 3; display: flex; flex-direction: column;
      background: #000; border-radius: 10px; overflow: hidden;
    }
    #video-panel .panel-title {
      padding: 6px 12px; background: #111; font-size: 0.85rem; color: #aaa;
    }
    #video-panel img { width: 100%; height: 100%; object-fit: contain; }

    /* ── Detection list panel ───────────────────────── */
    #list-panel {
      flex: 1; min-width: 220px; display: flex; flex-direction: column;
      background: #161616; border-radius: 10px; overflow: hidden;
      border: 1px solid #2a2a2a;
    }
    #list-panel .panel-title {
      padding: 10px 12px; background: #1a1a1a;
      font-size: 0.9rem; font-weight: bold;
      border-bottom: 1px solid #2a2a2a; display: flex;
      justify-content: space-between; align-items: center;
    }
    #list-panel .panel-title #det-count {
      font-size: 0.75rem; color: #4fc3f7; background: #0d2a3a;
      padding: 2px 8px; border-radius: 10px;
    }
    #detection-list { flex: 1; overflow-y: auto; padding: 8px; }
    #detection-list::-webkit-scrollbar { width: 5px; }
    #detection-list::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

    .det-item {
      padding: 7px 10px; margin-bottom: 5px; border-radius: 7px;
      background: #1e1e1e; font-size: 0.84rem;
      display: flex; justify-content: space-between; align-items: center;
      border-left: 3px solid #4fc3f7;
      animation: slideIn 0.25s ease;
    }
    .det-item .label { font-weight: bold; color: #4fc3f7; }
    .det-item .meta { text-align: right; }
    .det-item .conf { font-size: 0.8rem; color: #aaa; display: block; }
    .det-item .ts   { font-size: 0.72rem; color: #555; display: block; }

    .empty-msg { color: #444; text-align: center; margin-top: 40px; font-size: 0.85rem; }

    @keyframes slideIn {
      from { opacity: 0; transform: translateX(8px); }
      to   { opacity: 1; transform: translateX(0); }
    }
  </style>
</head>
<body>
  <header>
    🤖 <span>Hailo AI HAT</span>&nbsp;— Real-time Object Detection (YOLOv8 · COCO)
  </header>
  <main>
    <div id="video-panel">
      <div class="panel-title">📷 Live Camera Feed</div>
      <img src="/video_feed" alt="Live camera feed"/>
    </div>
    <div id="list-panel">
      <div class="panel-title">
        📋 Detected Objects
        <span id="det-count">0</span>
      </div>
      <div id="detection-list">
        <p class="empty-msg">Waiting for detections…</p>
      </div>
    </div>
  </main>

  <script>
    const listEl  = document.getElementById('detection-list');
    const countEl = document.getElementById('det-count');

    const src = new EventSource('/detections');

    src.onmessage = (e) => {
      const dets = JSON.parse(e.data);
      countEl.textContent = dets.length;

      if (!dets.length) {
        listEl.innerHTML = '<p class="empty-msg">No objects detected</p>';
        return;
      }

      listEl.innerHTML = dets.map(d =>
        `<div class="det-item">
           <span class="label">${d.label}</span>
           <span class="meta">
             <span class="conf">${d.confidence}%</span>
             <span class="ts">${d.time}</span>
           </span>
         </div>`
      ).join('');
    };

    src.onerror = () => { console.warn('SSE connection lost, retrying…'); };
  </script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(_INDEX_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        stream_with_context(_mjpeg_generator()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detections")
def detections_sse():
    return Response(
        stream_with_context(_sse_generator()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hailo AI HAT – Real-time object detection web app"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=(
            "Path to HEF model file. "
            "Defaults to yolov8s_h8l.hef (Hailo-8L) or yolov8m_h10.hef (Hailo-10H)."
        ),
    )
    parser.add_argument(
        "-l", "--labels",
        default="coco.txt",
        help="Path to class labels file (default: coco.txt).",
    )
    parser.add_argument(
        "-s", "--score_thresh",
        type=float,
        default=0.5,
        help="Detection confidence threshold, 0–1 (default: 0.5).",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="Web server port (default: 8080).",
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Picamera2 camera index (default: 0).",
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = (
            "/usr/share/hailo-models/yolov8m_h10.hef"
            if hailo_architecture() == "HAILO10H"
            else "/usr/share/hailo-models/yolov8s_h8l.hef"
        )

    print(f"Model  : {args.model}")
    print(f"Labels : {args.labels}")
    print(f"Thresh : {args.score_thresh}")
    print(f"Camera : {args.camera}")
    print(f"Server : http://0.0.0.0:{args.port}")

    # Start inference in a background daemon thread
    t = threading.Thread(
        target=inference_loop,
        args=(args.model, args.labels, args.score_thresh, args.camera),
        daemon=True,
    )
    t.start()

    app.run(host="0.0.0.0", port=args.port, threaded=True)
