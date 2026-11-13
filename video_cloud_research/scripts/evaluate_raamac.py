"""Đánh giá RAAMAC bằng cloud dựng lại từ video, dùng pipeline chính đã đóng băng.

Ba bước: coverage v3 (plane, 0.1/0.2, 2000 mẫu, seed 42) → config ngữ nghĩa v5
đóng băng → so khớp cohort đóng băng (validation + test). KHÔNG tune gì ở đây.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
IFC = ROOT / "data" / "schependomlaan" / "raw" / "model.ifc"
TRANSFORM = ROOT / "artifacts" / "week26_registration_v3" / "candidate_transform.json"
CONFIG = ROOT / "artifacts" / "cohort_v1" / "validation_semantic_config_v5.json"
COHORT = ROOT / "artifacts" / "cohort_v1"
TYPES = ["IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof",
         "IfcBeam", "IfcColumn", "IfcCovering"]


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def subset(pred_path, labels_path, out_path):
    with open(labels_path, newline="", encoding="utf-8-sig") as h:
        ids = {r["sample_id"] for r in csv.DictReader(h)}
    with open(pred_path, newline="", encoding="utf-8-sig") as h:
        reader = csv.DictReader(h)
        fields = reader.fieldnames
        rows = [r for r in reader if r["sample_id"] in ids]
    if {r["sample_id"] for r in rows} != ids:
        raise ValueError("subset mismatch")
    with open(out_path, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cloud = args.aligned_dir / "recon_aligned.ply"
    attrs = args.aligned_dir / "recon_attributes.npz"

    run([PY, ROOT / "progress_baseline.py",
         "--ifc", IFC, "--cloud", cloud, "--attributes", attrs,
         "--transform", TRANSFORM, "--scan-id", "week26",
         "--types", *TYPES,
         "--output", args.out_dir / "baseline.csv",
         "--samples", "2000", "--distance", "0.1", "--lateral", "0.2",
         "--seed", "42", "--workers", "2"])
    run([PY, ROOT / "apply_semantic.py",
         "--predictions", args.out_dir / "baseline.csv",
         "--config", CONFIG, "--ifc", IFC,
         "--attributes", attrs, "--transform", TRANSFORM,
         "--output", args.out_dir / "predictions.csv"])

    summary = {}
    for split, labels in (("validation", COHORT / "validation_labels.csv"),
                          ("test", COHORT / "test_labels_frozen.csv")):
        sub = args.out_dir / f"predictions_{split}.csv"
        subset(args.out_dir / "predictions.csv", labels, sub)
        run([PY, ROOT / "evaluate_progress.py",
             "--labels", labels, "--predictions", sub,
             "--split", split, "--output", args.out_dir / f"metrics_{split}.json"])
        m = json.loads((args.out_dir / f"metrics_{split}.json").read_text())
        pc = m.get("per_class", {})
        summary[split] = {
            "accuracy": m["accuracy"], "n_correct": m["n_correct"],
            "n_samples": m["n_samples"], "macro_f1": m["macro_f1"],
            "wilson95": m["accuracy_wilson_95_interval"],
            "majority_baseline_accuracy": m.get("majority_baseline_accuracy"),
            "beats_majority_baseline": m.get("beats_majority_baseline"),
            "built_recall": pc.get("built", {}).get("recall"),
            "partial_recall": pc.get("partial", {}).get("recall"),
        }

    print("\n===== KẾT QUẢ CLOUD DỰNG LẠI TỪ VIDEO (trung thực) =====", flush=True)
    for split, s in summary.items():
        print(f"{split}: {s['n_correct']}/{s['n_samples']} = {s['accuracy']:.4f} "
              f"macroF1={s['macro_f1']:.4f}", flush=True)
        print(f"   built recall={s['built_recall']}  partial recall={s['partial_recall']}  "
              f"baseline đa số={s['majority_baseline_accuracy']}  "
              f"thắng baseline={s['beats_majority_baseline']}", flush=True)
    if not summary["test"]["beats_majority_baseline"]:
        print("LƯU Ý: accuracy bằng/dưới baseline đoán not_built hàng loạt — "
              "CHƯA thực sự phát hiện tiến độ (built recall 0).", flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()