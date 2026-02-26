#!/usr/bin/env python3
"""
AI Hat - Object Detection with Hailo 8L
Realtime object detection với Hailo AI accelerator và web interface
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import threading
import time
from datetime import datetime
import os
import queue
from collections import deque, Counter
import supervision as sv

# Import Hailo utilities (đã copy ra thư mục gốc)
from utils import HailoAsyncInference

# ── Configuration ──────────────────────────────────────────────────────────────
HEF_PATH = "yolov8n.hef"          # Hailo model (đổi sang contrung_model.hef nếu muốn)
LABEL_PATH = "contrung_labels.txt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5

# ── Flask app ───────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Global state ────────────────────────────────────────────────────────────────
global_frame = None
frame_lock = threading.Lock()
current_detections = []          # detections của frame hiện tại
detections_lock = threading.Lock()
detection_history = deque(maxlen=200)   # lịch sử
fps_queue = deque(maxlen=30)

# ── Load labels ─────────────────────────────────────────────────────────────────
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# ── Hailo init ──────────────────────────────────────────────────────────────────
input_queue = queue.Queue()
output_queue = queue.Queue()
hailo_inference = HailoAsyncInference(
    hef_path=HEF_PATH,
    input_queue=input_queue,
    output_queue=output_queue,
)
threading.Thread(target=hailo_inference.run, daemon=True).start()

# ── Supervision tracking ─────────────────────────────────────────────────────────
tracker = sv.ByteTrack()
box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()


# ───────────────────────────────────────────────────────────────────────────────
# Camera capture thread
# ───────────────────────────────────────────────────────────────────────────────
def camera_capture_loop(index):
    global global_frame
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được camera index {index}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Không đọc được frame, thử lại...")
            time.sleep(0.5)
            continue
        with frame_lock:
            global_frame = frame.copy()
        time.sleep(0.03)
    cap.release()


def get_latest_frame():
    with frame_lock:
        return global_frame.copy() if global_frame is not None else None


# ───────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ───────────────────────────────────────────────────────────────────────────────
def extract_detections(hailo_output, h, w, threshold=0.5):
    xyxy, confidence, class_id = [], [], []
    for i, detections in enumerate(hailo_output):
        if len(detections) == 0:
            continue
        for detection in detections:
            bbox, score = detection[:4], detection[4]
            if score < threshold:
                continue
            bbox[0], bbox[1], bbox[2], bbox[3] = (
                bbox[1] * w, bbox[0] * h, bbox[3] * w, bbox[2] * h,
            )
            xyxy.append(bbox)
            confidence.append(score)
            class_id.append(i)
    return {
        "xyxy": np.array(xyxy, dtype=np.float32).reshape(-1, 4),
        "confidence": np.array(confidence, dtype=np.float32),
        "class_id": np.array(class_id, dtype=np.int32),
    }


def process_frame(frame):
    global current_detections
    model_h, model_w, _ = hailo_inference.get_input_shape()
    resized = cv2.resize(frame, (model_w, model_h))
    input_queue.put([resized])
    _, hailo_results = output_queue.get()

    if len(hailo_results) == 1:
        hailo_results = hailo_results[0]

    det = extract_detections(hailo_results, frame.shape[0], frame.shape[1],
                             threshold=CONFIDENCE_THRESHOLD)

    sv_det = sv.Detections(
        xyxy=det["xyxy"],
        confidence=det["confidence"],
        class_id=det["class_id"],
    )
    sv_det = tracker.update_with_detections(sv_det)

    labels = []
    objects = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for cls_id, trk_id, conf, box in zip(
        sv_det.class_id, sv_det.tracker_id, sv_det.confidence, sv_det.xyxy
    ):
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        x1, y1, x2, y2 = box
        objects.append({
            "id": int(trk_id) if trk_id is not None else None,
            "class": name,
            "confidence": round(float(conf), 3),
            "bbox": [round(float(x1), 1), round(float(y1), 1),
                     round(float(x2), 1), round(float(y2), 1)],
            "timestamp": now,
        })
        labels.append(f"#{trk_id} {name} {conf:.2f}")

    # Update global current detections
    with detections_lock:
        current_detections = objects

    # Log to history
    if objects:
        detection_history.append({
            "timestamp": now,
            "objects": objects,
            "count": len(objects),
        })

    # Annotate
    annotated = box_annotator.annotate(scene=frame.copy(), detections=sv_det)
    annotated = label_annotator.annotate(scene=annotated, detections=sv_det, labels=labels)
    return annotated, objects


# ───────────────────────────────────────────────────────────────────────────────
# Video stream generator
# ───────────────────────────────────────────────────────────────────────────────
def video_stream_generator():
    prev_time = time.time()
    while True:
        frame = get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        annotated_frame, detected_objects = process_frame(frame)

        # FPS calc
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        fps_queue.append(fps)
        avg_fps = sum(fps_queue) / len(fps_queue)
        prev_time = now

        # Overlay info
        h, w = annotated_frame.shape[:2]
        cv2.putText(
            annotated_frame,
            f"FPS: {avg_fps:.1f}  |  Objects: {len(detected_objects)}  |  {w}x{h}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
        )

        _, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)


# ───────────────────────────────────────────────────────────────────────────────
# Flask routes
# ───────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('ai_hat.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/current')
def api_current():
    """Detections trong frame hiện tại (polling realtime)"""
    with detections_lock:
        data = list(current_detections)
    avg_fps = round(sum(fps_queue) / len(fps_queue), 2) if fps_queue else 0.0
    return jsonify({"fps": avg_fps, "objects": data})


@app.route('/api/stats')
def api_stats():
    """Thống kê tổng hợp"""
    counts = Counter()
    for entry in detection_history:
        for obj in entry.get("objects", []):
            counts[obj["class"]] += 1
    avg_fps = round(sum(fps_queue) / len(fps_queue), 2) if fps_queue else 0.0
    return jsonify({
        "avg_fps": avg_fps,
        "total_events": len(detection_history),
        "object_counts": dict(counts),
        "recent": list(detection_history)[-30:],
    })


@app.route('/api/history')
def api_history():
    return jsonify(list(detection_history))


# ───────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs("logs", exist_ok=True)
    os.makedirs("captures", exist_ok=True)

    print("=" * 55)
    print("  AI Hat - Hailo 8L Object Detection")
    print("=" * 55)
    print(f"  Model   : {HEF_PATH}")
    print(f"  Labels  : {LABEL_PATH}  ({len(class_names)} classes)")
    print(f"  Camera  : index {CAMERA_INDEX}")
    print(f"  Thresh  : {CONFIDENCE_THRESHOLD}")
    print(f"  Web     : http://0.0.0.0:5000")
    print("=" * 55)

    threading.Thread(target=camera_capture_loop, args=(CAMERA_INDEX,), daemon=True).start()
    time.sleep(2)  # Chờ camera khởi động
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

