#!/usr/bin/env bash
# get_hailo_models.sh
#
# Sao chép (từ /usr/share/hailo-models/) hoặc tải về tất cả các file HEF
# của Hailo-8L example vào thư mục hailo_models/.
#
# Cách dùng:
#   chmod +x get_hailo_models.sh
#   ./get_hailo_models.sh
#
# Nếu hệ thống không có sẵn /usr/share/hailo-models/, script sẽ tải file từ
# Hailo Model Zoo (yêu cầu kết nối Internet).

set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")" && pwd)/hailo_models"
SYSTEM_MODEL_DIR="/usr/share/hailo-models"

# Base URL cho Hailo Model Zoo (compiled for Hailo-8L v2.14)
BASE_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l"

# Danh sách các model HEF Hailo-8L cần lấy
MODELS=(
    "yolov8s_h8l.hef"
    "yolov8n_h8l.hef"
    "yolov5m_wo_spp_60p.hef"
    "yolov5s_personface.hef"
    "yolov8s_pose_h8l.hef"
    "yolov5n_seg.hef"
    "ssd_mobilenet_v1.hef"
    "resnet_v1_50.hef"
    "scrfd_10g.hef"
)

mkdir -p "$DEST_DIR"

echo "=== Hailo-8L Model Downloader ==="
echo "Thư mục đích: $DEST_DIR"
echo ""

for filename in "${MODELS[@]}"; do
    dest_path="$DEST_DIR/$filename"

    if [[ -f "$dest_path" ]]; then
        echo "[SKIP]     $filename (đã tồn tại)"
        continue
    fi

    # Thử copy từ thư mục hệ thống trước
    if [[ -f "$SYSTEM_MODEL_DIR/$filename" ]]; then
        echo "[COPY]     $filename (từ $SYSTEM_MODEL_DIR)"
        cp "$SYSTEM_MODEL_DIR/$filename" "$dest_path"
        continue
    fi

    # Nếu không có trong hệ thống, tải từ Hailo S3
    url="$BASE_URL/$filename"
    echo "[DOWNLOAD] $filename"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest_path" "$url" || {
            echo "  ✗ Không tải được $filename" >&2
            rm -f "$dest_path"
        }
    elif command -v curl &>/dev/null; then
        curl -fL --progress-bar -o "$dest_path" "$url" || {
            echo "  ✗ Không tải được $filename" >&2
            rm -f "$dest_path"
        }
    else
        echo "  ✗ Cần cài wget hoặc curl để tải model" >&2
    fi
done

echo ""
echo "=== Hoàn thành ==="
echo "Các file HEF hiện có trong $DEST_DIR:"
ls -lh "$DEST_DIR"/*.hef 2>/dev/null || echo "  (chưa có file nào – kiểm tra log lỗi ở trên)"
