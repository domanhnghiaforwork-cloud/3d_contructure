"""Pipeline TỰ ĐỘNG một lệnh end-to-end: video/frame -> dựng 3D (Colab GPU)
-> căn chỉnh vào cloud tham chiếu (rigid scale=1 cho MASt3R metric, scale-search
cho DUSt3R non-metric) -> ước lượng normals -> đánh giá RAAMAC trên cohort đóng
băng -> báo cáo trung thực per-class + baseline đa số (beats_majority_baseline).

Một lệnh duy nhất điều phối toàn bộ; khâu dựng tự quản lý session Colab
(new->upload->run->download->STOP). --skip-remote dùng fused.ply đã có để
chạy lại phần căn chỉnh+đánh giá cục bộ (không tốn GPU).

URL chính thức: Colab endpoint Google; remote_{dust3r,mast3r}.py chỉ clone
github.com/naver/{dust3r,mast3r} + HF naver. Không proxy.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "bin" / "python"
ART = SCRIPTS.parent / "artifacts"
REF_ATTRS = ROOT / "artifacts/week26_raw_diagnostics/cloud_attributes.npz"


def run(cmd, stream=True):
    print("\n+", " ".join(str(c) for c in cmd), flush=True)
    if stream:
        subprocess.run([str(c) for c in cmd], check=True)
    else:
        subprocess.run([str(c) for c in cmd], check=True,
                       capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dust3r", choices=["dust3r", "mast3r"])
    ap.add_argument("--graph", default="logwin")
    ap.add_argument("--niter", default="1000")
    ap.add_argument("--conf", default="1.5")
    ap.add_argument("--stride", default="2")
    ap.add_argument("--size", default="384")
    ap.add_argument("--session", default=None)
    ap.add_argument("--out-dir", type=Path, default=ART / "pipeline_run")
    ap.add_argument("--skip-remote", action="store_true",
                    help="bỏ qua Colab, dùng out-dir/fused.ply đã có")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fused = args.out_dir / "fused.ply"
    if not args.skip_remote:
        cmd = [PY, SCRIPTS / "run_dust3r_colab.py",
               "--backend", args.backend, "--graph", args.graph,
               "--niter", args.niter, "--conf", args.conf,
               "--stride", args.stride, "--size", args.size,
               "--out-dir", args.out_dir]
        if args.session:
            cmd += ["--session", args.session]
        run(cmd)
    if not fused.exists():
        raise SystemExit(f"Thiếu cloud dựng lại: {fused}")

    align_dir = args.out_dir / "alignment_ref"
    norm_dir = args.out_dir / "alignment_ref_normals"
    eval_dir = args.out_dir / "eval"

    align_cmd = [PY, SCRIPTS / "align_and_compare.py",
                 "--recon", fused, "--ref-attributes", REF_ATTRS,
                 "--out-dir", align_dir]
    align_cmd.append("--rigid" if args.backend == "mast3r" else "--scale-search")
    run(align_cmd)
    run([PY, SCRIPTS / "estimate_normals.py",
         "--aligned-dir", align_dir, "--out-dir", norm_dir])
    run([PY, SCRIPTS / "evaluate_raamac.py",
         "--aligned-dir", norm_dir, "--out-dir", eval_dir])

    summary = json.loads((eval_dir / "summary.json").read_text())
    t = summary["test"]
    report = {
        "backend": args.backend, "skip_remote": args.skip_remote,
        "test": t, "validation": summary["validation"],
    }
    (args.out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=1))
    print("\n===== PIPELINE TỰ ĐỘNG: KẾT QUẢ (trung thực) =====", flush=True)
    print(f"backend={args.backend}  test={t['n_correct']}/{t['n_samples']}="
          f"{t['accuracy']:.4f}  val={summary['validation']['accuracy']:.4f}", flush=True)
    print(f"built recall={t.get('built_recall')}  partial recall={t.get('partial_recall')}  "
          f"baseline đa số={t.get('majority_baseline_accuracy')}  "
          f"thắng baseline={t.get('beats_majority_baseline')}", flush=True)
    if not t.get("beats_majority_baseline"):
        print("LƯU Ý: accuracy bằng/dưới baseline đoán not_built hàng loạt — "
              "CHƯA thực sự phát hiện tiến độ (built recall 0).", flush=True)
    print("report ->", args.out_dir / "pipeline_report.json", flush=True)


if __name__ == "__main__":
    main()