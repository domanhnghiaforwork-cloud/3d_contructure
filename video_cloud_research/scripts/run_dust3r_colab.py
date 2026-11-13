"""Orchestrator DUSt3R tự động: new T4 -> upload frames -> chạy remote_dust3r
-> tải fused.ply -> STOP (finally). Tham số dựng qua env (graph/niter/conf/stride).

Chỉ dùng URL chính thức trong remote_dust3r.py (github.com/naver/dust3r, HF
naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt). Colab CLI dùng endpoint Google.
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
COLAB = os.environ.get("COLAB_BIN") or shutil.which("colab") or str(Path.home() / ".local/bin/colab")
ART = SCRIPTS.parent / "artifacts"


def colab(*args, check=True, timeout=None):
    cmd = [COLAB, "--auth=oauth2", *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, timeout=timeout,
                          capture_output=True, text=True)


def exec_script(session, code, timeout=0, stream=False):
    tmp = Path(f"/tmp/vcr_exec_{int(time.time()*1000)%1000000}.py")
    tmp.write_text(code)
    cmd = [COLAB, "--auth=oauth2", "exec", "-s", session, "-f", str(tmp),
           "--timeout", str(timeout)]
    print("+", " ".join(cmd), flush=True)
    wall = timeout + 120 if timeout else None
    if stream:
        r = subprocess.run(cmd, timeout=wall)
        if r.returncode != 0:
            raise RuntimeError(f"exec failed rc={r.returncode}")
        return r
    r = subprocess.run(cmd, check=False, timeout=wall,
                       capture_output=True, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.stderr.strip():
        print("[stderr]", r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"exec failed rc={r.returncode}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=f"dust{int(time.time())%100000}")
    ap.add_argument("--backend", default="dust3r", choices=["dust3r", "mast3r"])
    ap.add_argument("--graph", default="logwin")
    ap.add_argument("--niter", default="3000")
    ap.add_argument("--conf", default="1.5")
    ap.add_argument("--stride", default="2")
    ap.add_argument("--size", default="384")
    ap.add_argument("--out-dir", type=Path, default=ART / "dust3r_hi")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    s = args.session
    remote_script = f"remote_{args.backend}.py"
    fused_name = f"fused_{args.backend}.ply"
    stats_name = f"stats_{args.backend}.json"

    colab("new", "-s", s, "--gpu", "T4", timeout=900)
    try:
        exec_script(s, "from pathlib import Path\n"
                       "Path('/content/vcr/out').mkdir(parents=True, exist_ok=True)\n"
                       "print('BOOT OK')\n", timeout=120)
        chunks = sorted(ART.glob("f4chunk_*"))
        if not chunks:
            subprocess.run(["split", "-b", "60m", "-d",
                            str(ART / "frames_4fps.zip"), str(ART / "f4chunk_")], check=True)
            chunks = sorted(ART.glob("f4chunk_*"))
        for ch in chunks:
            colab("upload", "-s", s, str(ch), f"/content/vcr/{ch.name}", timeout=900)
        reassemble_code = (
            "from pathlib import Path\n"
            "ws=Path('/content/vcr')\n"
            "zp=ws/'frames_4fps.zip'\n"
            "chunks=sorted(ws.glob('f4chunk_*'))\n"
            "if chunks:\n"
            "    with open(zp,'wb') as out:\n"
            "        for ch in chunks: out.write(ch.read_bytes())\n"
            "    for ch in chunks: ch.unlink()\n"
            "print('zip size', zp.stat().st_size)\n")
        for attempt in range(3):
            try:
                exec_script(s, reassemble_code, timeout=600)
                break
            except subprocess.TimeoutExpired:
                print(f"reassemble timeout, retry {attempt + 1}/3", flush=True)
        else:
            raise RuntimeError("reassemble failed after 3 attempts")
        colab("upload", "-s", s, str(SCRIPTS / remote_script),
              f"/content/vcr/{remote_script}", timeout=300)
        wrapper = (
            "import os, runpy\n"
            f"os.environ['VCR_GRAPH']='{args.graph}'\n"
            f"os.environ['VCR_NITER']='{args.niter}'\n"
            f"os.environ['VCR_CONF']='{args.conf}'\n"
            f"os.environ['VCR_STRIDE']='{args.stride}'\n"
            f"os.environ['VCR_SIZE']='{args.size}'\n"
            "os.environ['VCR_RECOMPUTE']='1'\n"
            f"runpy.run_path('/content/vcr/{remote_script}', run_name='__main__')\n")
        exec_script(s, wrapper, timeout=1800, stream=True)
        colab("download", "-s", s, f"/content/vcr/out/{fused_name}",
              str(args.out_dir / "fused.ply"), timeout=900)
        colab("download", "-s", s, f"/content/vcr/out/{stats_name}",
              str(args.out_dir / "stats.json"), timeout=300)
        print("DOWNLOAD OK ->", args.out_dir, flush=True)
    finally:
        colab("stop", "-s", s, check=False, timeout=300)
        print("SESSION STOPPED", s, flush=True)


if __name__ == "__main__":
    main()