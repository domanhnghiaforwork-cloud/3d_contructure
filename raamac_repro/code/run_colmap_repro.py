"""Orchestrator COLMAP tái hiện RAAMAC: new T4 -> upload frames -> remote_colmap
-> tải fused.ply + stats.json -> STOP (finally). Một lệnh cho một tuần.

Dùng Colab CLI chính thức (endpoint Google). Frame 1080p 5fps đã trích sẵn.
Chạy: python run_colmap_repro.py --week 26
"""
import argparse
import os
import subprocess
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
COLAB = os.environ.get("COLAB_CLI", "/home/adminpc/.local/bin/colab")
FRAMES = REPRO / "frames"
MASK = REPRO / "data" / "watermark_mask.png"
REMOTE = HERE / "remote_colmap.py"


def colab(*args, check=True, timeout=None):
    cmd = [COLAB, "--auth=oauth2", *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, timeout=timeout,
                          capture_output=True, text=True)


def exec_script(session, code, timeout=0, stream=False):
    tmp = Path(f"/tmp/repro_exec_{int(time.time()*1000)%1000000}.py")
    tmp.write_text(code)
    cmd = [COLAB, "--auth=oauth2", "exec", "-s", session, "-f", str(tmp),
           "--timeout", str(timeout)]
    print("+", " ".join(cmd), flush=True)
    wall = timeout + 180 if timeout else None
    if stream:
        r = subprocess.run(cmd, timeout=wall)
        if r.returncode != 0:
            raise RuntimeError(f"exec failed rc={r.returncode}")
        return r
    r = subprocess.run(cmd, check=False, timeout=wall, capture_output=True, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.stderr.strip():
        print("[stderr]", r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"exec failed rc={r.returncode}")
    return r


def build_zip(week, out_dir):
    src = FRAMES / f"week{week}"
    jpgs = sorted(src.glob("*.jpg"))
    zp = out_dir / "frames.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for j in jpgs:
            z.write(j, f"frames/{j.name}")
    print(f"zip {zp.name}: {len(jpgs)} frames, {zp.stat().st_size/1e6:.1f} MB", flush=True)
    return zp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True, choices=[26, 27, 28, 29, 30])
    ap.add_argument("--session", default=None)
    ap.add_argument("--mvs-size", default="1200")
    ap.add_argument("--sift-size", default="1920")
    ap.add_argument("--overlap", default="10")
    ap.add_argument("--exec-timeout", type=int, default=14400)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (REPRO / "artifacts" / f"week{args.week}")
    out_dir.mkdir(parents=True, exist_ok=True)
    s = args.session or f"repro{args.week}_{int(time.time())%100000}"

    zp = build_zip(args.week, out_dir)
    chunks = sorted(out_dir.glob("fchunk_*"))
    for ch in chunks:
        ch.unlink()
    subprocess.run(["split", "-b", "60m", "-d", str(zp), str(out_dir / "fchunk_")], check=True)
    chunks = sorted(out_dir.glob("fchunk_*"))

    colab("new", "-s", s, "--gpu", "T4", timeout=900)
    try:
        exec_script(s, "from pathlib import Path\n"
                       "Path('/content/vcr/out').mkdir(parents=True, exist_ok=True)\n"
                       "print('BOOT OK')\n", timeout=120)
        for ch in chunks:
            colab("upload", "-s", s, str(ch), f"/content/vcr/{ch.name}", timeout=900)
        reassemble = (
            "from pathlib import Path\n"
            "ws=Path('/content/vcr'); zp=ws/'frames.zip'\n"
            "chunks=sorted(ws.glob('fchunk_*'))\n"
            "with open(zp,'wb') as out:\n"
            "    for ch in chunks: out.write(ch.read_bytes())\n"
            "for ch in chunks: ch.unlink()\n"
            "print('zip size', zp.stat().st_size)\n")
        for attempt in range(3):
            try:
                exec_script(s, reassemble, timeout=600)
                break
            except subprocess.TimeoutExpired:
                print(f"reassemble timeout, retry {attempt+1}/3", flush=True)
        else:
            raise RuntimeError("reassemble failed")
        colab("upload", "-s", s, str(MASK), "/content/vcr/watermark_mask.png", timeout=300)
        colab("upload", "-s", s, str(REMOTE), "/content/vcr/remote_colmap.py", timeout=300)
        wrapper = (
            "import os, runpy\n"
            "os.environ['VCR_ZIP']='frames.zip'\n"
            f"os.environ['VCR_MVS_SIZE']='{args.mvs_size}'\n"
            f"os.environ['VCR_SIFT_SIZE']='{args.sift_size}'\n"
            f"os.environ['VCR_OVERLAP']='{args.overlap}'\n"
            "runpy.run_path('/content/vcr/remote_colmap.py', run_name='__main__')\n")
        exec_script(s, wrapper, timeout=args.exec_timeout, stream=True)
        check = exec_script(s,
            "from pathlib import Path\n"
            "import json\n"
            "p=Path('/content/vcr/out/fused.ply')\n"
            "if not p.exists():\n"
            "    raise SystemExit('REMOTE RECONSTRUCTION FAILED: no fused.ply')\n"
            "print('FUSED OK', p.stat().st_size)\n"
            "print(Path('/content/vcr/out/stats.json').read_text())\n",
            timeout=120)
        if "FUSED OK" not in (check.stdout or ""):
            raise RuntimeError("Remote reconstruction failed (no FUSED OK marker)")
        colab("download", "-s", s, "/content/vcr/out/fused.ply",
              str(out_dir / "fused.ply"), timeout=1200)
        colab("download", "-s", s, "/content/vcr/out/stats.json",
              str(out_dir / "stats.json"), timeout=300)
        print("DOWNLOAD OK ->", out_dir, flush=True)
    finally:
        colab("stop", "-s", s, check=False, timeout=300)
        print("SESSION STOPPED", s, flush=True)


if __name__ == "__main__":
    main()