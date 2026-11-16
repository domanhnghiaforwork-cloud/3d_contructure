"""Đối chiếu kết quả pipeline với danh sách phần tử thiếu của RAAMAC (mỗi tuần).

LƯU Ý QUAN TRỌNG: file RAAMAC dùng IFC as-planned (Synchro export) có GlobalID
KHÁC với design model.ifc của pipeline -> KHÔNG khớp GUID trực tiếp được.
Script này đối chiếu THEO TÊN (bỏ hậu tố '(#NNN)') và báo cáo phạm vi:
- Nhiều phần tử RAAMAC thiếu là cửa sổ (IfcWindow) NGOÀI 7 loại IFC pipeline đánh giá.
- Đánh giá per-element chuẩn vẫn là cohort đóng băng (evaluate_raamac.py).
Đây là cross-check phụ, xấp xỉ, không phải ground-truth độc lập.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TARGET = HERE.parent / "results" / "raamac_missing_target.json"
base = lambda n: re.sub(r"_\(#\d+\)$", "", n or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    target = json.loads(args.target.read_text())[str(args.week)]
    by_name = defaultdict(list)
    for r in csv.DictReader(args.predictions.open(encoding="utf-8-sig")):
        by_name[base(r["name"])].append(r["prediction"])

    rows = []
    in_scope = 0
    agree = 0
    for e in target:
        bn = base(e["name"])
        preds = by_name.get(bn, [])
        scope = bool(preds)
        if scope:
            in_scope += 1
            nb_frac = sum(p == "not_built" for p in preds) / len(preds)
            if nb_frac >= 0.5:
                agree += 1
        else:
            nb_frac = None
        rows.append({"name": e["name"], "base_name": bn,
                     "in_pipeline_scope": scope,
                     "n_elements_same_name": len(preds),
                     "our_not_built_fraction": round(nb_frac, 3) if nb_frac is not None else None})

    report = {
        "week": args.week,
        "note": "Name-based approximate cross-check; RAAMAC xlsx uses a different IFC GUID space. "
                "Authoritative per-element eval = frozen cohort (evaluate_raamac.py).",
        "raamac_missing_total": len(target),
        "in_pipeline_scope": in_scope,
        "out_of_scope": len(target) - in_scope,
        "in_scope_agree_not_built": agree,
        "elements": rows,
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print("saved", args.out)


if __name__ == "__main__":
    main()