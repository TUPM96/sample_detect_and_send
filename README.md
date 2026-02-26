# Sample Detect and Send

Real-time object detection web app chạy trên **Raspberry Pi 5 + Hailo AI HAT**, dùng YOLOv8 / COCO qua camera Pi.  
Web UI hiển thị live video (MJPEG) kèm danh sách object được phát hiện theo thời gian thực.

---

## Cấu trúc thư mục

```
sample_detect_and_send/
├── ai_hat.py              # Flask web server + Hailo inference loop (COCO)
├── utils.py               # HailoAsyncInference helper class
├── gps.py                 # Script test GPS qua serial NMEA
├── config.json            # Hardware config (motor speed, timing…)
├── con_trung/             # Submodule: insect detection dataset/scripts
├── data/
│   ├── coco.txt           # 80 COCO class labels
│   └── contrung_labels.txt  # 9 insect class labels
├── templates/
│   └── index.html         # Web UI (live video + detection list)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

> **Model HEF**: file `.hef` không được commit lên git.  
> Copy file vào cùng thư mục với `ai_hat.py` (xem hướng dẫn bên dưới).

---

## Phần cứng yêu cầu

| Thành phần | Ghi chú |
|---|---|
| Raspberry Pi 5 | 4 GB RAM trở lên |
| Hailo AI HAT+ (8L hoặc 10H) | Gắn trực tiếp lên header GPIO |
| Pi Camera (Module 3 khuyến nghị) | Kết nối qua CSI ribbon cable |
| Thẻ SD ≥ 32 GB | Raspberry Pi OS Bookworm (64-bit) |
| GPS module (tùy chọn) | Kết nối qua UART (`/dev/ttyAMA0`) – dùng với `gps.py` |

---

## 1. Kết nối Raspberry Pi qua VNC

### Cài VNC Viewer trên máy tính

Tải **RealVNC Viewer** tại <https://www.realvnc.com/en/connect/download/viewer/>

### Thông tin kết nối

| | |
|---|---|
| **Địa chỉ IP** | IP của Raspberry Pi trên mạng nội bộ (xem trên router hoặc dùng `arp -a`) |
| **Username** | `pi5` |
| **Password** | `pi5` |

### Các bước

1. Mở VNC Viewer → nhập địa chỉ IP Raspberry Pi → **Connect**
2. Nhập username `pi5`, password `pi5` → **OK**
3. Desktop Raspberry Pi hiển thị trên màn hình máy tính
4. Mở **Terminal** (Ctrl + Alt + T hoặc click icon terminal)

---

## 2. Lần đầu: Cài đặt

```bash
# Clone repository
git clone https://github.com/TUPM96/sample_detect_and_send.git
cd sample_detect_and_send

# (Tùy chọn) tạo virtual env
python3 -m venv .venv && source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

### Đặt model HEF

Copy file `.hef` phù hợp vào thư mục gốc của project:

```bash
# Hailo-8L
cp /usr/share/hailo-models/yolov8s_h8l.hef .

# hoặc Hailo-10H
cp /usr/share/hailo-models/yolov8m_h10.hef .
```

Script tự tìm file `.hef` theo thứ tự:
1. `yolov8s_h8l.hef` / `yolov8m_h10.hef` (theo board) trong cùng thư mục
2. `model.hef` trong cùng thư mục
3. `/usr/share/hailo-models/` (system path)

> **Model côn trùng**: dùng `contrung_model.hef` + `data/contrung_labels.txt` nếu muốn phát hiện côn trùng:
> ```bash
> python3 ai_hat.py -m contrung_model.hef -l data/contrung_labels.txt
> ```

---

## 3. Chạy test GPS

Dùng `gps.py` để kiểm tra module GPS độc lập (không cần Hailo):

```bash
python3 gps.py
```

**Output mẫu:**

```
Waiting for fix... sats=4 fix=0 rmc=V
FIX OK lat=10.762622 lon=106.660172 sats=8 fix=1 rmc=A
FIX OK lat=10.762623 lon=106.660173 sats=8 fix=1 rmc=A
```

> GPS phải kết nối vào `/dev/ttyAMA0` (UART0).  
> Cần bật Serial port trong `raspi-config` → Interface Options → Serial Port.

---

## 4. Chạy toàn bộ ứng dụng (ai_hat.py)

```bash
python3 ai_hat.py
```

### Tùy chọn dòng lệnh

```
python3 ai_hat.py [-m MODEL] [-l LABELS] [-s SCORE] [-p PORT] [-c CAMERA]

  -m, --model       Đường dẫn file HEF (mặc định: tự tìm trong thư mục)
  -l, --labels      File class labels (mặc định: data/coco.txt)
  -s, --score_thresh Ngưỡng confidence 0–1 (mặc định: 0.5)
  -p, --port        Port web server (mặc định: 8080)
  -c, --camera      Chỉ số camera Picamera2 (mặc định: 0)
```

**Ví dụ:**

```bash
# Chạy mặc định trên port 8080
python3 ai_hat.py

# Chỉ định rõ model và hạ ngưỡng confidence
python3 ai_hat.py -m yolov8s_h8l.hef -s 0.4

# Chạy trên port 5000
python3 ai_hat.py -p 5000
```

### Truy cập Web UI

Sau khi khởi động, mở trình duyệt trên **cùng mạng nội bộ**:

```
http://<IP_RASPBERRY_PI>:8080
```

Hoặc ngay trên Raspberry Pi:

```
http://localhost:8080
```

**Giao diện gồm:**
- **Bên trái**: Live video từ camera với bounding box
- **Bên phải**: Danh sách object đang detect (tên, confidence %, giờ)
- **Header**: Tên model HEF đang dùng + số FPS

---

## 5. Chạy cả object detection lẫn GPS

Mở 2 terminal song song trong VNC:

```bash
# Terminal 1 – khởi động web detection
cd ~/sample_detect_and_send
python3 ai_hat.py

# Terminal 2 – theo dõi GPS
cd ~/sample_detect_and_send
python3 gps.py
```

---

## 6. Khởi động tự động khi bật nguồn (tùy chọn)

```bash
# Tạo systemd service
sudo nano /etc/systemd/system/ai-hat.service
```

Nội dung file:

```ini
[Unit]
Description=Hailo AI HAT Object Detection
After=network.target

[Service]
User=pi5
WorkingDirectory=/home/pi5/sample_detect_and_send
ExecStart=/usr/bin/python3 ai_hat.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

> **Lưu ý**: Thay `/home/pi5/sample_detect_and_send` bằng đường dẫn thực tế nơi bạn clone project.
> Kiểm tra bằng lệnh `pwd` khi đang ở trong thư mục project.

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-hat.service
sudo systemctl start ai-hat.service

# Kiểm tra trạng thái
sudo systemctl status ai-hat.service
```

---

## Troubleshooting

| Lỗi | Nguyên nhân | Xử lý |
|---|---|---|
| `ModuleNotFoundError: picamera2` | Chưa cài picamera2 | `pip install picamera2` |
| `No cameras available` | Camera chưa bật | `raspi-config` → Interface → Camera → Enable |
| `HailoRT device not found` | Hailo HAT chưa nhận | Kiểm tra kết nối PCIe, chạy `hailortcli fw-control identify` |
| `FileNotFoundError: *.hef` | Model chưa có | Copy file `.hef` vào thư mục project (xem mục 2) |
| `Permission denied: /dev/ttyAMA0` | UART chưa bật | `raspi-config` → Interface → Serial Port → Enable |
| Web UI không hiển thị | Port bị chặn | Kiểm tra firewall: `sudo ufw allow 8080` |
