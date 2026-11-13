"""Orchestrator pipeline tự động: video/frame -> Colab (SfM+MVS) -> căn chỉnh
-> đánh giá RAAMAC trên cohort đóng băng -> báo cáo mốc 50/60/70/80.

Quản lý vòng đời session: new -> upload -> run -> download -> STOP (finally).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "bin" / "python"
COLAB = "/home/adminpc/.local/bin/colab"
ART = SCRIPTS.parent / "artifacts"


def colab(*args, check=True, timeout=None):
    cmd = [COLAB, "--auth=oauth2", *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, timeout=timeout,
                          capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--session", default=f"auto{int(time.time()) % 100000}")
    ap.add_argument("--frames-zip", type=Path, default=ART / "frames_4fps.zip")
    ap.add_argument("--mvs-size", default="1600")
    ap.add_argument("--out-dir", type=Path, default=ART / "auto_run1")
    ap.add_argument("--skip-remote", action="store_true",
                    help="bỏ qua Colab, dùng fused.ply đã có trong out-dir")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    s = args.session

    if not args.skip_remote:
        gpu = args.gpu
        r = colab("new", "-s", s, "--gpu", gpu, check=False, timeout=900)
        if r.returncode != 0 or "READY" not in r.stdout + r.stderr:
            print(f"{gpu} unavailable, fallback T4", flush=True)
            gpu = "T4"
            colab("new", "-s", s, "--gpu", gpu, timeout=900)
        try:
            colab("exec", "-s", s, "-f", "/tmp/vcr_boot3.py", timeout=300)
            chunks = sorted(ART.glob("f4chunk_*"))
            if not chunks:
                subprocess.run(["split", "-b", "60m", "-d", str(args.frames_zip),
                                str(ART / "f4chunk_")], check=True)
                chunks = sorted(ART.glob("f4chunk_*"))
            for ch in chunks:
                colab("upload", "-s", s, str(ch), f"/content/vcr/{ch.name}", timeout=900)
            for f in ("watermark_mask.png",):
                colab("upload", "-s", s, str(ART / f), f"/content/vcr/{f}", timeout=300)
            for f in ("colmap_reconstruct.py", "dense_mvs.py", "remote_full.py"):
                colab("upload", "-s", s, str(SCRIPTS / f), f"/content/vcr/{f}", timeout=300)
            join = (
                "import subprocess, glob\n"
                "parts = sorted(glob.glob('/content/vcr/f4chunk_*'))\n"
                f"subprocess.run('cat ' + ' '.join(parts) + ' > /content/vcr/{args.frames_zip.name}', shell=True, check=True)\n"
                "print('joined', len(parts))\n"
            )
            (Path("/tmp") / "vcr_join_auto.py").write_text(join)
            colab("exec", "-s", s, "-f", "/tmp/vcr_join_auto.py", timeout=300)
            colab("exec", "--timeout", "0", "--env", f"VCR_ZIP={args.frames_zip.name}",
                  "--env", f"VCR_MVS_SIZE={args.mvs_size}", "-s", s,
                  "-f", str(SCRIPTS / "remote_full.py"))
            colab("download", "-s", s, "/content/vcr/out/best_model.txt",
                  str(args.out_dir / "best_model.txt"), timeout=300)
            best = (args.out_dir / "best_model.txt").read_text().strip()
            colab("download", "-s", s, f"/content/vcr/out/fused_{best}.ply",
                  str(args.out_dir / "fused.ply"), timeout=1800)
            colab("download", "-s", s, f"/content/vcr/out/stats_{best}.json",
                  str(args.out_dir / "stats.json"), timeout=300)
        finally:
            colab("stop", "-s", s, check=False, timeout=300)
            print(f"session {s} stopped", flush=True)

    print("\n########## ALIGN + GATE ##########", flush=True)
    align_dir = args.out_dir / "alignment"
    r = subprocess.run([str(PY), str(SCRIPTS / "align_to_bim.py"),
                        "--recon", str(args.out_dir / "fused.ply"),
                        "--out-dir", str(align_dir)], check=False)
    if r.returncode != 0:
        print("QUALITY GATE FAILED — reconstruction distorted; consider DUSt3R fallback", flush=True)
        sys.exit(2)

    print("\n########## EVALUATE RAAMAC ##########", flush=True)
    eval_dir = args.out_dir / "eval"
    subprocess.run([str(PY), str(SCRIPTS / "evaluate_raamac.py"),
                    "--aligned-dir", str(align_dir), "--out-dir", str(eval_dir)], check=True)
    summary = json.loads((eval_dir / "summary.json").read_text())
    acc = summary["test"]["accuracy"]
    report = {"test_accuracy": acc, "milestones": {
        f"{m:.0%}": acc >= m for m in (0.5, 0.6, 0.7, 0.8)}, "summary": summary}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()