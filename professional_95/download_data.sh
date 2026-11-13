#!/usr/bin/env bash
# Tải dữ liệu gốc NẶNG (week26.ply + model.ifc) từ nguồn công khai chính thức.
# Hai file này bị .gitignore trong repo để giữ repo nhẹ; chạy script sau khi clone.
# Nguồn: https://github.com/openBIMstandards/Archive-DataSetSchependomlaan
set -euo pipefail
cd "$(dirname "$0")"
DEST=data/raw
mkdir -p "$DEST"
SRC=https://github.com/openBIMstandards/Archive-DataSetSchependomlaan
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "Clone nguồn dữ liệu chính thức: $SRC"
git clone --depth 1 "$SRC" "$TMP/dataset"
IFC=$(find "$TMP/dataset" -iname '*.ifc' | head -1 || true)
PLY=$(find "$TMP/dataset" -iname 'week26*.ply' | head -1 || true)
if [ -n "$IFC" ]; then cp -v "$IFC" "$DEST/model.ifc"; else echo "CẢNH BÁO: không tìm thấy .ifc trong $SRC"; fi
if [ -n "$PLY" ]; then cp -v "$PLY" "$DEST/week26.ply"; else echo "CẢNH BÁO: không tìm thấy week26*.ply trong $SRC"; fi
echo "Ho tất. Nội dung $DEST:"
ls -la "$DEST"