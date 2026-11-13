#!/usr/bin/env bash
# Chạy lại đánh giá RAAMAC nhánh chuyên nghiệp từ data có sẵn trong package.
# Cần: python3 + numpy/scipy/trimesh/ifcopenshell/sklearn; data/raw/{week26.ply,model.ifc}
# (nếu thiếu raw: bash download_data.sh). Kết quả: out/metrics_test.json (kỳ vọng 95.24%).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p out
TYPES=(IfcWall IfcWallStandardCase IfcSlab IfcRoof IfcBeam IfcColumn IfcCovering)

"$PY" code/progress_baseline.py --ifc data/raw/model.ifc \
  --cloud data/raw/week26.ply --attributes data/week26_raw_diagnostics/cloud_attributes.npz \
  --transform data/week26_registration_v3/candidate_transform.json --scan-id week26 \
  --types "${TYPES[@]}" --output out/baseline.csv \
  --samples 2000 --distance 0.1 --lateral 0.2 --seed 42 --workers 2

"$PY" code/apply_semantic.py --predictions out/baseline.csv \
  --config data/cohort_v1/validation_semantic_config_v5.json --ifc data/raw/model.ifc \
  --attributes data/week26_raw_diagnostics/cloud_attributes.npz \
  --transform data/week26_registration_v3/candidate_transform.json --output out/predictions.csv

for split in validation test; do
  "$PY" - "$split" <<'PY'
import csv, sys
split = sys.argv[1]
lab = {"validation": "data/cohort_v1/validation_labels.csv",
       "test": "data/cohort_v1/test_labels_frozen.csv"}[split]
with open(lab, newline="", encoding="utf-8-sig") as h:
    ids = {r["sample_id"] for r in csv.DictReader(h)}
with open("out/predictions.csv", newline="", encoding="utf-8-sig") as h:
    rd = csv.DictReader(h); fields = rd.fieldnames
    rows = [r for r in rd if r["sample_id"] in ids]
assert {r["sample_id"] for r in rows} == ids, "subset mismatch"
with open(f"out/predictions_{split}.csv", "w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
PY
  labels=data/cohort_v1/validation_labels.csv
  [ "$split" = test ] && labels=data/cohort_v1/test_labels_frozen.csv
  "$PY" code/evaluate_progress.py --labels "$labels" \
    --predictions "out/predictions_${split}.csv" --split "$split" \
    --output "out/metrics_${split}.json"
done
echo "Ho tất -> out/metrics_test.json (kỳ vọng 20/21 = 95.24%)"