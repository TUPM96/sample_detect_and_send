#!/usr/bin/env python3
"""
ai_hat.py – Insect detection web app using Hailo AI HAT.

Rewritten based on app_realtime.py architecture:
  - HailoAsyncInference (utils.py) for inference
  - cv2.VideoCapture for camera
  - ByteTrack tracking via supervision
  - MQTT for publishing detections and controlling hardware
  - Detection logging (CSV + image capture)
  - Conveyor auto-control based on detected insects
  - LED / UVA light control via MQTT
  - FPS + resolution overlay on video stream

⚠  This server binds to 0.0.0.0 and is intended for use on a trusted local
   network only.  Do not expose it directly to the public internet.

Usage:
    python ai_hat.py [-m MODEL] [-l LABELS] [-s SCORE] [-p PORT] [-c CAMERA]
"""

import argparse
import glob
import json
import os
import queue
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
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

CAPTURE_DIR = os.path.join(_HERE, "captures")
LOG_DIR = os.path.join(_HERE, "logs")
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CONFIG_FILE = os.path.join(_HERE, "config.json")
DEFAULT_CONFIG = {"speed": 255, "time": 1000, "auto_stop_delay": 5}

# ---------------------------------------------------------------------------
# MQTT settings
# ---------------------------------------------------------------------------

MQTT_HOST = "103.146.22.13"
MQTT_PORT = 1883
MQTT_USER = "user1"
MQTT_PASS = "12345678"
MQTT_TOPIC = "doan/contrung/control"
SENSOR_TOPIC = "doan/contrung/sensor"
MQTT_SCHEDULE_RESP = "doan/contrung/schedule"
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
schedule_cache: list = []

logged_tracker_ids: set = set()
conveyor_running: bool = False
last_insect_time: float = time.time()
_conveyor_lock = threading.Lock()   # protects conveyor_running / last_insect_time / logged_tracker_ids
# Populated after CLI arg parsing
input_queue: queue.Queue = None
output_queue: queue.Queue = None
hailo_inference: HailoAsyncInference = None
class_names: list = []

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        _write_config(DEFAULT_CONFIG)
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def _write_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------


def camera_capture_loop(camera_index: int) -> None:
    global global_frame
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Không mở được camera index {camera_index}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không lấy được frame, dừng thread camera.")
            break
        with frame_lock:
            global_frame = frame.copy()
        time.sleep(0.03)
    cap.release()


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

    Returns (annotated_frame, insects_list).
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
    insects_list = []
    labels = []
    for cid, tid, box in zip(sv_dets.class_id, sv_dets.tracker_id, sv_dets.xyxy):
        name = class_names[cid]
        x1, y1, x2, y2 = box
        width_mm = round(abs(x2 - x1) * PIXEL_TO_MM, 2)
        height_mm = round(abs(y2 - y1) * PIXEL_TO_MM, 2)
        insects_list.append({
            "tracker_id": int(tid) if tid is not None else None,
            "class": name,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "detected_at": now,
        })
        labels.append(f"#{tid} {name} {width_mm}x{height_mm}mm")

    annotated = box_annotator.annotate(scene=frame.copy(), detections=sv_dets)
    annotated = label_annotator.annotate(scene=annotated, detections=sv_dets, labels=labels)
    return annotated, insects_list

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


def send_detect_mqtt(insects_list: list) -> None:
    _mqtt_publish({"insects": insects_list}, MQTT_DETECT_RESULT)


def send_conveyor_control(speed: int, time_ms: int) -> None:
    _mqtt_publish({"speed": speed, "time": time_ms})

# ---------------------------------------------------------------------------
# Detection logging
# ---------------------------------------------------------------------------


def log_detection(dt: datetime, counts: Counter, image_path: str, detect_id: str) -> None:
    log_file = os.path.join(LOG_DIR, f"detect_{dt.strftime('%Y-%m-%d')}.csv")
    total = sum(counts.values())
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{detect_id},{dt.strftime('%Y-%m-%d %H:%M:%S')},{total},"
            f"{json.dumps(dict(counts), ensure_ascii=False)},{image_path}\n"
        )

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


def _on_schedule_resp(client, userdata, msg):
    global schedule_cache
    try:
        data = json.loads(msg.payload.decode())
        if isinstance(data, list):
            schedule_cache = data
    except Exception:
        pass


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
    threading.Thread(target=_run, args=(MQTT_SCHEDULE_RESP, _on_schedule_resp), daemon=True).start()

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
        global conveyor_running, last_insect_time
        frame_count = 0
        last_fps_time = time.time()
        fps = 0.0
        resolution = None

        while True:
            frame = _get_latest_frame()
            if frame is not None:
                frame_draw, insects_list = process_frame(frame.copy(), score_thresh)

                # Auto conveyor control (protected by lock)
                now = time.time()
                cfg = load_config()
                auto_stop_delay = cfg.get("auto_stop_delay", 5)
                with _conveyor_lock:
                    if insects_list:
                        last_insect_time = now
                        if not conveyor_running:
                            send_conveyor_control(cfg.get("speed", 185), -1)  # -1 = run continuously
                            conveyor_running = True
                            print("Bắt đầu băng tải vì phát hiện côn trùng")
                    else:
                        if conveyor_running and (now - last_insect_time > auto_stop_delay):
                            send_conveyor_control(0, 0)
                            conveyor_running = False
                            print(f"Dừng băng tải sau {auto_stop_delay}s không phát hiện côn trùng")

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

                # MQTT – only send newly-seen tracker IDs (protected by lock)
                new_insects = []
                new_tids: set = set()
                for item in insects_list:
                    tid = item.get("tracker_id")
                    if tid is not None and tid not in new_tids:
                        with _conveyor_lock:
                            already_logged = tid in logged_tracker_ids
                        if not already_logged:
                            new_insects.append(item)
                            new_tids.add(tid)
                if new_insects:
                    counts = Counter(item["class"] for item in new_insects)
                    detect_id = str(uuid.uuid4())
                    image_path = os.path.join(CAPTURE_DIR, f"{detect_id}.jpg")
                    cv2.imwrite(image_path, frame_draw)
                    log_detection(datetime.now(), counts, image_path, detect_id)
                    send_detect_mqtt(new_insects)
                    with _conveyor_lock:
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


@app.route("/stat_counts")
def stat_counts():
    log_files = glob.glob(os.path.join(LOG_DIR, "detect_*.csv"))
    stat: dict = defaultdict(Counter)
    for file in log_files:
        try:
            with open(file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 5:
                        continue
                    day = parts[1].split(" ")[0]
                    try:
                        counts = json.loads(parts[3])
                        for k, v in counts.items():
                            stat[day][k] += int(v)
                    except Exception:
                        continue
        except Exception:
            continue
    days = sorted(stat.keys())
    all_types = sorted({k for ct in stat.values() for k in ct})
    colors = ["#f44336", "#2196f3", "#4caf50", "#ff9800", "#9c27b0", "#009688", "#e91e63", "#607d8b"]
    datasets = [
        {
            "label": t,
            "data": [stat[d].get(t, 0) for d in days],
            "backgroundColor": colors[i % len(colors)],
        }
        for i, t in enumerate(all_types)
    ]
    return jsonify({"labels": days, "datasets": datasets})


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


@app.route("/get_config")
def get_config():
    return jsonify(load_config())


@app.route("/save_config", methods=["POST"])
def save_config_api():
    try:
        data = request.get_json()
        cfg = {
            "speed": int(data.get("speed", 185)),
            "ddos": int(data.get("ddos", 5)),
            "auto_stop_delay": float(data.get("auto_stop_delay", 5)),
        }
        _write_config(cfg)
        return jsonify({"status": "ok", "msg": "Đã lưu cấu hình!"})
    except Exception as e:
        return jsonify({"status": "fail", "msg": str(e)}), 400


@app.route("/list_schedule")
def list_schedule():
    try:
        client = mqtt.Client()
        client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.connect(MQTT_HOST, MQTT_PORT, 60)
        client.publish(MQTT_TOPIC, json.dumps({"action": "get_schedule"}))
        client.disconnect()
    except Exception as e:
        print(f"list_schedule MQTT error: {e}")
    for _ in range(20):
        if schedule_cache:
            break
        time.sleep(0.1)
    return jsonify(schedule_cache)


def _led_action(payload: dict, ok_msg: str):
    try:
        client = mqtt.Client()
        client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.connect(MQTT_HOST, MQTT_PORT, 60)
        client.publish(MQTT_TOPIC, json.dumps(payload))
        client.disconnect()
        return jsonify({"msg": ok_msg})
    except Exception as e:
        return jsonify({"msg": f"Lỗi MQTT: {e}"}), 500


@app.route("/turn_on_led", methods=["POST"])
def turn_on_led():
    return _led_action({"led1": "on"}, "Đã bật đèn!")


@app.route("/turn_off_led", methods=["POST"])
def turn_off_led():
    return _led_action({"led1": "off"}, "Đã tắt đèn!")


@app.route("/turn_on_uva", methods=["POST"])
def turn_on_uva():
    return _led_action({"led2": "on"}, "Đã bật đèn UVA!")


@app.route("/turn_off_uva", methods=["POST"])
def turn_off_uva():
    return _led_action({"led2": "off"}, "Đã tắt đèn UVA!")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hailo AI HAT – Insect detection web app"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Path to HEF model file (default: auto-detect).",
    )
    parser.add_argument(
        "-l", "--labels",
        default=os.path.join(_HERE, "data", "contrung_labels.txt"),
        help="Path to class labels file (default: data/contrung_labels.txt).",
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
    print(f"Labels : {args.labels}")
    print(f"Thresh : {args.score_thresh}")
    print(f"Camera : {args.camera}")
    print(f"Server : http://0.0.0.0:{args.port}")

    # Load class labels
    with open(args.labels, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f]

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

    # Stop conveyor at startup
    send_conveyor_control(0, 0)

    # Start camera capture thread
    threading.Thread(target=camera_capture_loop, args=(args.camera,), daemon=True).start()

    app.run(host="0.0.0.0", port=args.port, threaded=True)
