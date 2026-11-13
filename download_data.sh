#!/usr/bin/env bash
# Tải dữ liệu gốc NẶNG từ nguồn công khai chính thức (không commit vào repo).
# Nguồn dataset: https://github.com/openBIMstandards/Archive-DataSetSchependomlaan
# Cần: model.ifc (BIM) + week26.ply (cloud chuyên nghiệp) đặt tại data/schependomlaan/raw/.
# Trọng số DUSt3R/MASt3R KHÔNG tải ở đây — chúng tự tải từ Hugging Face trên Colab khi chạy.
set -euo pipefail
cd "$(dirname "$0")"
DEST=data/schependomlaan/raw
mkdir -p "$DEST"
SRC=https://github.com/openBIMstandards/Archive-DataSetSchependomlaan
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "Clone nguồn dữ liệu chính thức: $SRC"
git clone --depth 1 "$SRC" "$TMP/dataset"
echo "Tìm file IFC và cloud week26 trong dataset..."
IFC=$(find "$TMP/dataset" -iname '*.ifc' | head -1 || true)
PLY=$(find "$TMP/dataset" -iname 'week26*.ply' | head -1 || true)
if [ -n "$IFC" ]; then cp -v "$IFC" "$DEST/model.ifc"; else echo "CẢNH BÁO: không tìm thấy .ifc — kiểm tra thủ công trong $SRC"; fi
if [ -n "$PLY" ]; then cp -v "$PLY" "$DEST/week26.ply"; else echo "CẢNH BÁO: không tìm thấy week26*.ply — kiểm tra thủ công trong $SRC"; fi
echo "Ho tất. Nội dung $DEST:"
ls -la "$DEST"
echo "Nếu tên file khác, đổi thủ công về model.ifc / week26.ply cho đúng đường dẫn pipeline."