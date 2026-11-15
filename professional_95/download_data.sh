#!/usr/bin/env bash
# Tải dữ liệu gốc NẶNG còn thiếu (week27-30.ply, và week26.ply/model.ifc nếu thiếu)
# từ nguồn công khai chính thức release 1.1, giải nén đúng đường dẫn + đổi tên.
# Các ply tuần khác >100MB nên KHÔNG commit trong repo (xem data/raw/README.md).
# Nguồn: https://github.com/openBIMstandards/Archive-DataSetSchependomlaan (release 1.1)
set -euo pipefail
cd "$(dirname "$0")"
DEST=data/raw
mkdir -p "$DEST"
BASE=https://github.com/openBIMstandards/Archive-DataSetSchependomlaan/releases/download/1.1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

need_pc=0
for w in 26 27 28 29 30; do
  [ -f "$DEST/week$w.ply" ] || need_pc=1
done
if [ "$need_pc" = 1 ]; then
  echo "Tải Point.Clouds.zip (~743MB) từ nguồn chính thức..."
  curl -L --fail -o "$TMP/pc.zip" "$BASE/schependomlaan.-.Point.Clouds.zip"
  for w in 26 27 28 29 30; do
    [ -f "$DEST/week$w.ply" ] && { echo "bo qua week$w.ply (da co)"; continue; }
    if [ "$w" = 26 ]; then
      src="Point Clouds/Week 26/PLY/De Schepen week 27_t.ply"   # basename week27 trong thu muc Week26
    else
      src="Point Clouds/Week $w/PLY/De Schepen week ${w}_t.ply"
    fi
    echo "giai nen week$w.ply ..."
    unzip -p "$TMP/pc.zip" "$src" > "$DEST/week$w.ply"
  done
else
  echo "Du week26-30.ply, bo qua tai Point.Clouds.zip"
fi

if [ ! -f "$DEST/model.ifc" ]; then
  echo "Tải IFC.zip (~86MB) từ nguồn chính thức..."
  curl -L --fail -o "$TMP/ifc.zip" "$BASE/schependomlaan.-.IFC.zip"
  unzip -p "$TMP/ifc.zip" "Design model IFC/IFC Schependomlaan.ifc" > "$DEST/model.ifc"
else
  echo "Da co model.ifc, bo qua tai IFC.zip"
fi

echo "Ho tất. Nội dung $DEST:"
ls -la "$DEST"
echo "Kiem tra toan ven: sha256sum -c theo bang trong data/raw/README.md"