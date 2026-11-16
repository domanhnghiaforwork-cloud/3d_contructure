"""TỰ KIỂM TRA HÌNH HỌC một tuần sau khi dựng: căn chỉnh -> render screenshot -> metrics.

Chạy cục bộ CPU (không GPU). Deliverable là file .ply để NGƯỜI XEM tự đánh giá trực quan;
KHÔNG dùng đánh giá nhãn làm tiêu chí (map căn chỉnh chưa đạt -> eval nhãn gây hiểu lầm).

Hai đường vào, tự nhận diện theo nội dung artifacts/week{N}:
  - Có fused_shot*.ply  -> merge_shots.py (căn từng shot + hợp nhất -> week{N}_final.ply).
  - Chỉ có fused.ply     -> align_and_compare.py (căn một cloud vào hệ tham chiếu).
Sau đó render 4 góc (thô + đã căn chỉnh chồng tham chiếu) và ghi verify_summary.json (hình học).

Đánh giá tiến độ theo nhãn (cohort đóng băng) KHÔNG nằm ở đây — xem thư mục
`professional_95/` của repo cho nhánh đánh giá RAAMAC chuẩn trên cloud chuyên nghiệp.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
PY = os.environ.get("PYTHON") or sys.executable
ALIGN = HERE / "align_and_compare.py"
REF_ATTRS = PKG / "data" / "cloud_attributes.npz"
REF_PLY = PKG / "data" / "raw" / "week26.ply"


def run(cmd):
    print("\n+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rigid", action="store_true",
                    help="khóa scale=1 (chỉ cho cloud metric; COLMAP thì KHÔNG)")
    args = ap.parse_args()
    out = args.out_dir or (PKG / "artifacts" / f"week{args.week}")
    fused = out / "fused.ply"
    shot_clouds = sorted(out.glob("fused_shot*.ply"))
    if not fused.exists() and not shot_clouds:
        raise SystemExit(f"Thiếu cloud trong {out} — chạy run_colmap_repro/shots trước")

    align = out / "alignment_ref"
    if shot_clouds:
        run([PY, HERE / "merge_shots.py", "--week", str(args.week), "--out-dir", out])
    else:
        align_cmd = [PY, ALIGN, "--recon", fused, "--ref-attributes", REF_ATTRS,
                     "--out-dir", align]
        align_cmd.append("--rigid" if args.rigid else "--scale-search")
        run(align_cmd)

    raw_cloud = fused if fused.exists() else shot_clouds[0]
    run([PY, HERE / "render_cloud.py", "--cloud", raw_cloud,
         "--out", out / "renders_raw", "--title", f"week{args.week} RAW"])
    run([PY, HERE / "render_cloud.py",
         "--cloud", align / "recon_aligned.ply", "--ref", REF_PLY,
         "--out", out / "renders_aligned", "--title", f"week{args.week} ALIGNED"])

    metrics = json.loads((align / "metrics.json").read_text())
    final = out / f"week{args.week}_final.ply"
    deliverable = final if final.exists() else (align / "recon_aligned.ply")
    verify = {
        "week": args.week,
        "criterion": "geometry only — deliverable .ply để người xem tự đánh giá trực quan; "
                     "KHÔNG dùng eval nhãn làm tiêu chí (map chưa đạt).",
        "alignment": {
            "scale": metrics["scale"],
            "chamfer_mean_m": metrics["chamfer_mean_m"],
            "median_recon_to_ref_m": metrics["recon_to_ref_m"]["median"],
            "coverage_ref_at_0.2": metrics["coverage_ref_at_tau"]["0.2"],
            "n_recon": metrics["n_recon"],
        },
        "deliverable_ply": str(deliverable),
    }
    (out / "verify_summary.json").write_text(json.dumps(verify, indent=1, ensure_ascii=False))
    print("\n===== VERIFY SUMMARY (geometry) =====")
    print(json.dumps(verify, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
