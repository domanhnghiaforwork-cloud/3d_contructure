# Chạy lại tái hiện RAAMAC từ video drone -> point cloud .ply (mỗi tuần một bản).
# Mặc định CHỈ chạy bước hình học CPU (căn chỉnh + render + viewer) nếu đã có cloud dựng sẵn.
# Bước dựng COLMAP cần GPU Colab + video nguồn (tải bằng download_data.sh) -> bật bằng --recon.
#
# Cần: python3 + numpy, scipy, trimesh, matplotlib (+ OpenCV cho trích/detect frame).
#   PYTHON=/đường/dẫn/python3 bash reproduce.sh            # hình học tuần 26 (dùng cloud có sẵn)
#   PYTHON=... bash reproduce.sh --weeks 26 --recon        # dựng lại từ video trên Colab rồi merge
set -euo pipefail
cd "$(dirname "$0")"
PKG="$(pwd)"
PY="${PYTHON:-python3}"
WEEKS="26"
RECON=0
while [ $# -gt 0 ]; do
  case "$1" in
    --weeks) shift; WEEKS="$1";;
    --recon) RECON=1;;
    *) echo "Tham số lạ: $1" >&2; exit 2;;
  esac
  shift
done

if [ "$RECON" = "1" ]; then
  # Trích frame từ data/videos -> phát hiện cut -> dựng COLMAP từng shot trên Colab -> hợp nhất.
  "$PY" code/extract_all_frames.py
  "$PY" code/detect_shots.py
  for w in $WEEKS; do
    "$PY" code/run_colmap_shots.py --week "$w"
    "$PY" code/merge_shots.py --week "$w"
    "$PY" code/build_viewer_final.py --week "$w"
  done
else
  # Không tốn GPU: căn chỉnh cloud đã dựng (artifacts/week{N}/fused*.ply) + render tự kiểm tra.
  for w in $WEEKS; do
    "$PY" code/verify_week.py --week "$w"
  done
fi
echo "Xong. Kết quả: results/ + viewer/ (hình học); artifacts/week{N}/ (sản phẩm trung gian)."
