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
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, stream_with_context
from picamera2 import Picamera2
try:
    # picamera2 >= 0.3.21 exports hailo_architecture from the top-level devices package.
    from picamera2.devices import Hailo, hailo_architecture
except ImportError:
    # Older picamera2 versions only export Hailo from the sub-package.
    # Fall back to importing directly from the hailo sub-module.
    try:
        from picamera2.devices.hailo import Hailo, hailo_architecture
    except ImportError:
        # Last resort: Hailo class is available but hailo_architecture is not.
        # Define it locally by calling hailortcli directly.
        import subprocess
        from picamera2.devices.hailo import Hailo

        def hailo_architecture():
            """Detect Hailo device architecture via hailortcli."""
            try:
                out = subprocess.run(
                    ["hailortcli", "fw-control", "identify"],
                    capture_output=True, text=True, check=True,
                ).stdout
                for line in out.splitlines():
                    if "Device Architecture:" in line:
                        return line.split(":")[-1].strip()
            except (OSError, subprocess.CalledProcessError):
                pass
            return None

# Directory that contains this script – used for local HEF / label look-ups.
_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Shared state written by the inference thread, read by HTTP route handlers.
_lock = threading.Lock()
_latest_jpeg: bytes = b""
_current_detections: list = []
_model_path: str = ""

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
    global _latest_jpeg, _current_detections, _model_path
    _model_path = model_path

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
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info")
def info():
    return jsonify(model=os.path.basename(_model_path))


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
        default="data/coco.txt",
        help="Path to class labels file (default: data/coco.txt).",
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
        # 1. Check for a local HEF file placed next to this script,
        #    preferring the model that matches the detected Hailo architecture.
        arch = hailo_architecture()
        preferred = "yolov8m_h10.hef" if arch == "HAILO10H" else "yolov8s_h8l.hef"
        for local_name in (preferred, "model.hef"):
            local_path = os.path.join(_HERE, local_name)
            if os.path.isfile(local_path):
                args.model = local_path
                break
        # 2. Fall back to the system-installed Hailo models.
        if args.model is None:
            args.model = (
                "/usr/share/hailo-models/yolov8m_h10.hef"
                if arch == "HAILO10H"
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
