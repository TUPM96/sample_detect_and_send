# README_AI – Hailo-8L Sample Models (HEF Files)

Thư mục này chứa các file **HEF (Hailo Executable Format)** – định dạng model nhị phân
đã được biên dịch sẵn cho chip **Hailo-8L (8 TOPS)**.  
File `.hef` **không được commit lên git** (đã có trong `.gitignore`).  
Dùng script [`../get_hailo_models.sh`](../get_hailo_models.sh) để copy hoặc tải về.

```bash
cd ..
chmod +x get_hailo_models.sh
./get_hailo_models.sh
```

---

## Danh sách model mẫu

### 1. Object Detection – Phát hiện vật thể (COCO 80 lớp)

#### `yolov8s_h8l.hef` – YOLOv8 Small
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | YOLOv8s |
| Chip hỗ trợ | Hailo-8L (8 TOPS) |
| Input | 640 × 640, RGB |
| Dataset | COCO 2017 (80 lớp) |
| mAP@0.5:0.95 | ~37.4 |
| FPS (RPi5 + Hailo-8L) | ~30 FPS |

**Dùng với project này:**
```bash
python3 ai_hat.py -m hailo_models/yolov8s_h8l.hef -l data/coco.txt
```

---

#### `yolov8n_h8l.hef` – YOLOv8 Nano (nhanh nhất)
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | YOLOv8n |
| Chip hỗ trợ | Hailo-8L |
| Input | 640 × 640, RGB |
| Dataset | COCO 2017 (80 lớp) |
| mAP@0.5:0.95 | ~37.3 |
| FPS (RPi5 + Hailo-8L) | ~60+ FPS |

> **Nano** là model nhỏ nhất, tốc độ cao nhất, phù hợp khi cần FPS cao hơn là độ chính xác.

**Dùng với project này:**
```bash
python3 ai_hat.py -m hailo_models/yolov8n_h8l.hef -l data/coco.txt
```

---

#### `yolov5m_wo_spp_60p.hef` – YOLOv5 Medium (không SPP)
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | YOLOv5m (w/o SPP) |
| Chip hỗ trợ | Hailo-8L |
| Input | 640 × 640, RGB |
| Dataset | COCO 2017 (80 lớp) |
| mAP@0.5 | ~60 |

> Phiên bản YOLOv5 được Hailo tối ưu bỏ SPP block để tăng hiệu năng trên chip.

**Dùng với project này:**
```bash
python3 ai_hat.py -m hailo_models/yolov5m_wo_spp_60p.hef -l data/coco.txt
```

---

#### `ssd_mobilenet_v1.hef` – SSD MobileNetV1
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | SSD + MobileNetV1 |
| Chip hỗ trợ | Hailo-8L |
| Input | 300 × 300, RGB |
| Dataset | COCO 2017 (80 lớp) |
| mAP@0.5 | ~23.0 |

> Model rất nhẹ và nhanh. Phù hợp khi cần chạy liên tục ở FPS rất cao hoặc tài nguyên hạn chế.

---

### 2. Person & Face Detection – Phát hiện người và khuôn mặt

#### `yolov5s_personface.hef` – YOLOv5s Person & Face
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | YOLOv5s |
| Chip hỗ trợ | Hailo-8L |
| Input | 640 × 640, RGB |
| Lớp | 2: `person`, `face` |

> Chuyên phát hiện người và khuôn mặt, nhanh hơn khi chỉ cần 2 lớp.

---

#### `scrfd_10g.hef` – SCRFD 10G Face Detection
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | SCRFD-10GF |
| Chip hỗ trợ | Hailo-8L |
| Input | 640 × 640, RGB |
| Nhiệm vụ | Face detection + landmark (5 điểm) |
| WiderFace Easy/Med/Hard | 95.6 / 94.2 / 82.8 |

> Phát hiện khuôn mặt độ chính xác cao kèm 5 facial landmark (mắt trái, mắt phải, mũi, miệng trái, miệng phải).

---

### 3. Pose Estimation – Ước tính tư thế người

#### `yolov8s_pose_h8l.hef` – YOLOv8s Pose cho Hailo-8L
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | YOLOv8s-Pose |
| Chip hỗ trợ | Hailo-8L |
| Input | 640 × 640, RGB |
| Keypoints | 17 (COCO skeleton) |
| mAP_pose@0.5 | ~60.0 |

> Phát hiện người + vẽ 17 keypoint: đầu, vai, khuỷu tay, cổ tay, hông, gối, mắt cá chân.  
> Hữu ích cho ứng dụng gym tracker, an toàn lao động, trò chơi tương tác.

---

### 4. Instance Segmentation – Phân vùng từng đối tượng

#### `yolov5n_seg.hef` – YOLOv5n Segmentation
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | YOLOv5n-Seg |
| Chip hỗ trợ | Hailo-8L |
| Input | 640 × 640, RGB |
| Dataset | COCO 2017 (80 lớp) |
| mask mAP@0.5:0.95 | ~23.7 |

> Tạo **pixel mask** cho từng đối tượng, không chỉ bounding box.  
> Phù hợp khi cần biết chính xác hình dạng vật thể (ví dụ: cắt nền, đếm diện tích).

---

### 5. Image Classification – Phân loại ảnh

#### `resnet_v1_50.hef` – ResNet-50 v1
| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | ResNet-50 v1 |
| Chip hỗ trợ | Hailo-8L |
| Input | 224 × 224, RGB |
| Dataset | ImageNet ILSVRC2012 (1000 lớp) |
| Top-1 Acc | ~76.1 % |
| Top-5 Acc | ~92.9 % |

> Phân loại ảnh thành 1 trong 1000 danh mục ImageNet (chó, mèo, xe hơi, bàn phím…).  
> Không tạo bounding box – chỉ trả về nhãn và điểm số của toàn bộ ảnh.

---

## Tóm tắt nhanh

| File HEF | Nhiệm vụ | Chip | Input | Số lớp |
|---|---|---|---|---|
| yolov8s_h8l.hef | Object Detection | Hailo-8L | 640×640 | 80 |
| yolov8n_h8l.hef | Object Detection (Nano) | Hailo-8L | 640×640 | 80 |
| yolov5m_wo_spp_60p.hef | Object Detection | Hailo-8L | 640×640 | 80 |
| ssd_mobilenet_v1.hef | Object Detection (nhẹ) | Hailo-8L | 300×300 | 80 |
| yolov5s_personface.hef | Person & Face | Hailo-8L | 640×640 | 2 |
| scrfd_10g.hef | Face + Landmark | Hailo-8L | 640×640 | 1 |
| yolov8s_pose_h8l.hef | Pose Estimation | Hailo-8L | 640×640 | 1+17kp |
| yolov5n_seg.hef | Segmentation | Hailo-8L | 640×640 | 80 |
| resnet_v1_50.hef | Classification | Hailo-8L | 224×224 | 1000 |

---

## Chọn model phù hợp

| Yêu cầu | Model gợi ý |
|---|---|
| Phát hiện vật thể tổng quát | `yolov8s_h8l.hef` |
| Cần FPS tối đa, hy sinh độ chính xác | `yolov8n_h8l.hef` |
| Chỉ cần phát hiện người / khuôn mặt | `yolov5s_personface.hef` |
| Nhận diện tư thế / thể thao | `yolov8s_pose_h8l.hef` |
| Cần biết hình dạng vật thể (mask) | `yolov5n_seg.hef` |
| Phân loại ảnh (không cần vị trí) | `resnet_v1_50.hef` |
| Phát hiện khuôn mặt chi tiết + landmark | `scrfd_10g.hef` |

---

## Cách lấy file HEF

### Từ hệ thống (Raspberry Pi đã cài Hailo runtime)
```bash
ls /usr/share/hailo-models/
```

### Dùng script tự động
```bash
# Từ thư mục gốc của project
./get_hailo_models.sh
```

### Thủ công từ Hailo Model Zoo
```bash
# Ví dụ: tải yolov8s_h8l.hef
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov8s_h8l.hef \
     -P hailo_models/
```

---

## Lưu ý

- File `.hef` là định dạng **nhị phân riêng của Hailo** – không chỉnh sửa tay được.  
- Mỗi file được biên dịch cho **chip Hailo-8L**. Dùng trên chip khác sẽ báo lỗi khi chạy.  
- Phiên bản HEF phải **khớp với phiên bản HailoRT** đang cài trên Raspberry Pi.  
  Kiểm tra phiên bản: `hailortcli fw-control identify`  
- File `.hef` được git-ignore – không commit lên repository.
