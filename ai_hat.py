#!/usr/bin/env python3
"""
ai_hat.py – Object detection web app using Hailo AI HAT with COCO labels.

  - HailoAsyncInference (utils.py) for inference
  - cv2.VideoCapture for camera
  - ByteTrack tracking via supervision
  - MQTT for publishing detections
  - FPS + resolution overlay on video stream

⚠  This server binds to 0.0.0.0 and is intended for use on a trusted local
   network only.  Do not expose it directly to the public internet.

Usage:
    python ai_hat.py [-m MODEL] [-l LABELS] [-s SCORE] [-p PORT] [-c CAMERA]
"""

import argparse
import json
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import supervision as sv
from flask import Flask, Response, jsonify, render_template, request

from utils import HailoAsyncInference

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

PIXEL_TO_MM = 0.05

# ---------------------------------------------------------------------------
# COCO 80-class labels
# ---------------------------------------------------------------------------

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ---------------------------------------------------------------------------
# MQTT settings
# ---------------------------------------------------------------------------

MQTT_HOST = "103.146.22.13"
MQTT_PORT = 1883
MQTT_USER = "user1"
MQTT_PASS = "12345678"
MQTT_TOPIC = "doan/contrung/control"
SENSOR_TOPIC = "doan/contrung/sensor"
MQTT_DETECT_RESULT = "doan/contrung/result"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

global_frame = None
frame_lock = threading.Lock()

# supervision tracking (module-level so state persists across frames)
tracker = sv.ByteTrack()
box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()

sensor_data_buffer: deque = deque(maxlen=200)

logged_tracker_ids: set = set()
_tracker_lock = threading.Lock()   # protects logged_tracker_ids

# Populated after CLI arg parsing
input_queue: queue.Queue = None
output_queue: queue.Queue = None
hailo_inference: HailoAsyncInference = None
class_names: list = []

# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------


def _open_camera(camera_index: int):
    """Open the camera and return a VideoCapture, or None on failure."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback: let OpenCV choose the backend
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Drain the initial buffered frames so we get a live image
    for _ in range(5):
        cap.read()
    return cap


def _csi_camera_loop() -> None:
    """Capture frames using Picamera2 (CSI ribbon camera via libcamera)."""
    global global_frame
    print("Đang dùng camera CSI (Picamera2)...")
    try:
        from picamera2 import Picamera2  # noqa: PLC0415
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        try:
            while True:
                try:
                    frame = picam2.capture_array()
                    with frame_lock:
                        global_frame = frame.copy()
                except Exception as e:
                    print(f"Lỗi đọc frame CSI: {e}")
                time.sleep(0.03)
        finally:
            picam2.stop()
    except ImportError:
        print("Picamera2 không được cài đặt. Chạy: pip install picamera2")
    except Exception as e:
        print(f"Lỗi camera CSI: {e}")


def camera_capture_loop(camera_index: int) -> None:
    global global_frame
    MAX_CONSECUTIVE_FAILURES = 30   # ~1 s at 30 fps before reconnect attempt
    RECONNECT_DELAY = 3             # seconds between reconnect attempts
    MAX_RECONNECT_ATTEMPTS = 3      # fall back to CSI after this many failures

    cap = _open_camera(camera_index)
    if cap is None:
        print(f"Không mở được camera index {camera_index}")
        _csi_camera_loop()
        return

    failures = 0
    reconnect_attempts = 0
    while True:
        if cap is None:
            time.sleep(RECONNECT_DELAY)
            reconnect_attempts += 1
            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                _csi_camera_loop()
                return
            cap = _open_camera(camera_index)
            if cap is None:
                print(f"Không mở được camera index {camera_index}, thử lại sau {RECONNECT_DELAY}s...")
            continue

        ret, frame = cap.read()
        if ret:
            failures = 0
            reconnect_attempts = 0
            with frame_lock:
                global_frame = frame.copy()
            time.sleep(0.03)
        else:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"Camera index {camera_index}: quá nhiều lỗi liên tiếp, thử kết nối lại...")
                cap.release()
                cap = None
                failures = 0


def _get_latest_frame():
    with frame_lock:
        return global_frame.copy() if global_frame is not None else None

# ---------------------------------------------------------------------------
# Hailo detection + tracking
# ---------------------------------------------------------------------------


def extract_detections(hailo_output, h: int, w: int, threshold: float) -> dict:
    """Convert HailoRT postprocess output to arrays for supervision."""
    xyxy, confidence, class_id = [], [], []
    for i, detections in enumerate(hailo_output):
        if len(detections) == 0:
            continue
        for det in detections:
            bbox, score = det[:4], det[4]
            if score < threshold:
                continue
            # Hailo output: (y_min, x_min, y_max, x_max) normalised 0-1
            bbox[0], bbox[1], bbox[2], bbox[3] = (
                bbox[1] * w, bbox[0] * h,
                bbox[3] * w, bbox[2] * h,
            )
            xyxy.append(bbox)
            confidence.append(score)
            class_id.append(i)
    return {
        "xyxy": np.array(xyxy, dtype=np.float32).reshape(-1, 4),
        "confidence": np.array(confidence, dtype=np.float32),
        "class_id": np.array(class_id, dtype=np.int32),
    }


def process_frame(frame: np.ndarray, threshold: float):
    """Run Hailo inference + ByteTrack on one frame.

    Returns (annotated_frame, detections_list).
    """
    model_h, model_w, _ = hailo_inference.get_input_shape()
    input_frame = cv2.resize(frame, (model_w, model_h))
    input_queue.put([input_frame])
    _, hailo_results = output_queue.get()
    if len(hailo_results) == 1:
        hailo_results = hailo_results[0]

    dets = extract_detections(hailo_results, frame.shape[0], frame.shape[1], threshold)

    sv_dets = sv.Detections(
        xyxy=dets["xyxy"],
        confidence=dets["confidence"],
        class_id=dets["class_id"],
    )
    sv_dets = tracker.update_with_detections(sv_dets)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detections_list = []
    labels = []
    for cid, tid, box in zip(sv_dets.class_id, sv_dets.tracker_id, sv_dets.xyxy):
        name = class_names[cid]
        x1, y1, x2, y2 = box
        width_mm = round(abs(x2 - x1) * PIXEL_TO_MM, 2)
        height_mm = round(abs(y2 - y1) * PIXEL_TO_MM, 2)
        detections_list.append({
            "tracker_id": int(tid) if tid is not None else None,
            "class": name,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "detected_at": now,
        })
        labels.append(f"#{tid} {name} {width_mm}x{height_mm}mm")

    annotated = box_annotator.annotate(scene=frame.copy(), detections=sv_dets)
    annotated = label_annotator.annotate(scene=annotated, detections=sv_dets, labels=labels)
    return annotated, detections_list

# ---------------------------------------------------------------------------
# MQTT helpers
# ---------------------------------------------------------------------------


def _mqtt_publish(payload: dict, topic: str = MQTT_TOPIC) -> None:
    try:
        client = mqtt.Client()
        client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.connect(MQTT_HOST, MQTT_PORT, 60)
        client.publish(topic, json.dumps(payload, ensure_ascii=False))
        client.disconnect()
    except Exception as e:
        print(f"MQTT publish error: {e}")


def send_detect_mqtt(detections_list: list) -> None:
    _mqtt_publish({"detections": detections_list}, MQTT_DETECT_RESULT)

# ---------------------------------------------------------------------------
# MQTT subscribers
# ---------------------------------------------------------------------------


def _on_sensor_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        sensor_data_buffer.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "temperature": data.get("temperature"),
            "humidity": data.get("humidity"),
            "light": data.get("light"),
        })
    except Exception as e:
        print(f"Sensor MQTT parse error: {e}")


def start_mqtt_subscribers() -> None:
    def _run(topic, on_message):
        client = mqtt.Client()
        client.username_pw_set(MQTT_USER, MQTT_PASS)
        try:
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.subscribe(topic)
            client.on_message = on_message
            client.loop_forever()
        except Exception as e:
            print(f"MQTT subscriber [{topic}] error: {e}")

    threading.Thread(target=_run, args=(SENSOR_TOPIC, _on_sensor_message), daemon=True).start()

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera_stream")
@app.route("/video_feed")          # backward-compatible alias
def camera_stream():
    score_thresh = app.config.get("SCORE_THRESH", 0.5)

    def gen():
        frame_count = 0
        last_fps_time = time.time()
        fps = 0.0
        resolution = None

        while True:
            frame = _get_latest_frame()
            if frame is not None:
                frame_draw, detections_list = process_frame(frame.copy(), score_thresh)

                # FPS / resolution overlay
                frame_count += 1
                current_time = time.time()
                if resolution is None:
                    h, w = frame_draw.shape[:2]
                    resolution = f"{w}x{h}"
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    frame_count = 0
                    last_fps_time = current_time
                cv2.putText(
                    frame_draw,
                    f"Resolution: {resolution} | FPS: {fps:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA,
                )

                # MQTT – only send newly-seen tracker IDs
                new_detections = []
                new_tids: set = set()
                for item in detections_list:
                    tid = item.get("tracker_id")
                    if tid is not None and tid not in new_tids:
                        with _tracker_lock:
                            already_logged = tid in logged_tracker_ids
                        if not already_logged:
                            new_detections.append(item)
                            new_tids.add(tid)
                if new_detections:
                    send_detect_mqtt(new_detections)
                    with _tracker_lock:
                        logged_tracker_ids.update(new_tids)

                _, buf = cv2.imencode(".jpg", frame_draw)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/sensor_data")
def sensor_data():
    return jsonify(list(sensor_data_buffer))


@app.route("/control", methods=["POST"])
def control():
    try:
        data = request.get_json()
        _mqtt_publish(data)
        return jsonify({"status": "ok", "msg": "Đã gửi qua MQTT", "data": data})
    except Exception as e:
        return jsonify({"status": "fail", "msg": str(e)}), 400

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hailo AI HAT – COCO object detection web app"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Path to HEF model file (default: auto-detect).",
    )
    parser.add_argument(
        "-l", "--labels",
        default=None,
        help="Path to class labels file (default: built-in COCO 80 labels).",
    )
    parser.add_argument(
        "-s", "--score_thresh",
        type=float,
        default=0.5,
        help="Detection confidence threshold 0–1 (default: 0.5).",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=5000,
        help="Web server port (default: 5000).",
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture (default: 0).",
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model is None:
        candidates = [
            os.path.join(_HERE, "yolov8n.hef"),
            os.path.join(_HERE, "hailo_models", "yolov8s_h8l.hef"),
            "/usr/share/hailo-models/yolov8n.hef",
            "/usr/share/hailo-models/yolov8s_h8l.hef",
        ]
        for path in candidates:
            if os.path.isfile(path):
                args.model = path
                break
        if args.model is None:
            args.model = "/usr/share/hailo-models/yolov8s_h8l.hef"

    print(f"Model  : {args.model}")
    print(f"Labels : {args.labels or 'COCO 80 (built-in)'}")
    print(f"Thresh : {args.score_thresh}")
    print(f"Camera : {args.camera}")
    print(f"Server : http://0.0.0.0:{args.port}")

    # Load class labels
    if args.labels is not None:
        with open(args.labels, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f]
    else:
        class_names = COCO_LABELS

    # Initialise Hailo async inference
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    hailo_inference = HailoAsyncInference(
        hef_path=args.model,
        input_queue=input_queue,
        output_queue=output_queue,
    )
    threading.Thread(target=hailo_inference.run, daemon=True).start()

    # Store score threshold for use in routes
    app.config["SCORE_THRESH"] = args.score_thresh

    # Start MQTT subscribers (non-fatal if broker unreachable)
    start_mqtt_subscribers()

    # Start camera capture thread
    threading.Thread(target=camera_capture_loop, args=(args.camera,), daemon=True).start()

    app.run(host="0.0.0.0", port=args.port, threaded=True)
