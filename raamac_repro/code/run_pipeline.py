"""PIPELINE HOÀN CHỈNH tái hiện RAAMAC: video drone -> MỘT point cloud .ply hoàn thiện mỗi tuần.

Một lệnh chạy trọn (resume được), xuất deliverable là file .ply DUY NHẤT mỗi tuần:
  1. Trích frame 1080p 5fps (extract_all_frames.py).
  2. Phát hiện cut, chia shot (detect_shots.py) — video marketing ghép nhiều cảnh.
  3. Dựng COLMAP SfM+MVS RIÊNG từng shot trên Colab T4 (run_colmap_shots.py).
  4. Căn chỉnh từng shot vào hệ tham chiếu + HỢP NHẤT -> week{N}_final.ply (merge_shots.py).
  5. Dựng viewer một cloud (build_viewer_final.py) + render screenshot.

Chạy:  python run_pipeline.py                 # cả 5 tuần
       python run_pipeline.py --weeks 26      # chỉ tuần 26
       python run_pipeline.py --weeks 26 --skip-recon   # chỉ hợp nhất lại (đã có shot cloud)

KHÔNG dùng đánh giá nhãn làm tiêu chí (map chưa đúng -> eval nhãn sai lệch).
Deliverable = file .ply để người xem TỰ đánh giá trực quan. Metrics hình học chỉ tham khảo.
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
ART = PKG / "artifacts"


def run(cmd):
    print("\n+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, nargs="+", default=[26, 27, 28, 29, 30])
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-recon", action="store_true")
    ap.add_argument("--mvs-size", default="1200")
    ap.add_argument("--sift-size", default="1920")
    args = ap.parse_args()

    if not args.skip_extract:
        run([PY, HERE / "extract_all_frames.py"])
    run([PY, HERE / "detect_shots.py"])

    for w in args.weeks:
        out = ART / f"week{w}"
        if not args.skip_recon:
            if sorted(out.glob("fused_shot*.ply")):
                print(f"\n[week {w}] đã có shot cloud, bỏ qua dựng lại", flush=True)
            else:
                run([PY, HERE / "run_colmap_shots.py", "--week", str(w),
                     "--mvs-size", args.mvs_size, "--sift-size", args.sift_size])
        run([PY, HERE / "merge_shots.py", "--week", str(w)])
        run([PY, HERE / "build_viewer_final.py", "--week", str(w)])

    rows = []
    for w in args.weeks:
        ms = ART / f"week{w}" / "merge_summary.json"
        final = ART / f"week{w}" / f"week{w}_final.ply"
        if ms.exists():
            m = json.loads(ms.read_text())
            rows.append({"week": w, "final_ply": str(final),
                         "n_shots_merged": m["n_shots_merged"],
                         "n_points": m["n_points_merged"],
                         "coverage_ref_0.2": m.get("merged_coverage_ref_0.2"),
                         "chamfer_m": m.get("merged_chamfer_m")})
    report = {"deliverable": "Một file .ply hoàn thiện mỗi tuần (đã căn chỉnh, mét thật, đa shot). "
                             "Người xem tự đánh giá trực quan; không dùng nhãn làm tiêu chí.",
              "weeks": rows}
    (ART / "final_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("\n===== FINAL REPORT =====")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()