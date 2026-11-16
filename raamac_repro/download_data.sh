#!/usr/bin/env bash
# Tải dữ liệu nặng cho tái hiện RAAMAC (KHÔNG commit vào git vì quá lớn).
#   - Cloud/mô hình THAM CHIẾU chuyên nghiệp: week26.ply + model.ifc (copy từ nhánh
#     professional_95 đã có trong repo — tránh nhân bản 120MB).
#   - 5 video drone nguồn (1080p) cho bước dựng COLMAP: tải từ YouTube bằng yt-dlp.
# Kết quả vào data/raw/ và data/videos/. Sau đó chạy: bash reproduce.sh --recon
set -euo pipefail
cd "$(dirname "$0")"

# 1) Tham chiếu (dùng cho overlay viewer + căn chỉnh). Ưu tiên copy từ professional_95.
mkdir -p data/raw data/videos
REF_SRC="$(cd .. && pwd)/professional_95/data/raw"
if [ -f "$REF_SRC/week26.ply" ] && [ -f "$REF_SRC/model.ifc" ]; then
  cp -n "$REF_SRC/week26.ply" data/raw/ || true
  cp -n "$REF_SRC/model.ifc" data/raw/ || true
  echo "[ref] week26.ply + model.ifc <- professional_95/data/raw"
else
  echo "[ref] KHÔNG thấy $REF_SRC — lấy week26.ply/model.ifc từ bộ dữ liệu Schependomlaan"
  echo "      (TU Eindhoven, luận văn van Schaijk 2016) rồi đặt vào data/raw/."
fi

# 2) Video nguồn (cần yt-dlp: pip install -U yt-dlp). Mỗi tuần 1 chuyến bay.
if command -v yt-dlp >/dev/null 2>&1; then
  declare -A VID=( [26]=S0CGq1Ms3qQ [27]=_FA_8JYWoYQ [28]=gKLyBgRecLQ [29]=0Pev1Weh6_I [30]=QbBf31bY0xw )
  for w in "${!VID[@]}"; do
    out="data/videos/week${w}.mp4"
    [ -f "$out" ] && { echo "[vid] có sẵn week${w}.mp4"; continue; }
    yt-dlp -f 'bv*+ba/b' -o "$out" "https://www.youtube.com/watch?v=${VID[$w]}"
  done
  echo "[vid] xong data/videos/"
else
  echo "[vid] thiếu yt-dlp — cài: pip install -U yt-dlp, rồi chạy lại script này."
fi
