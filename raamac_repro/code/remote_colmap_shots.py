"""COLMAP đa shot trên Colab T4: mỗi shot video = một 'chuyến bay' riêng.

Video marketing bị cắt nhiều shot khác góc -> dựng chung vỡ model. Dựng riêng từng
shot (SfM+MVS), xuất fused_shot{id}.ply. Cục bộ sẽ căn chỉnh từng shot vào hệ tham
chiếu rồi HỢP NHẤT cloud đã căn chỉnh -> phủ đa góc (faithful RAAMAC đa chuyến bay).

Đọc shots.json (từ detect_shots.py). Resume theo marker mỗi shot. Bỏ qua shot lỗi.
Env: VCR_MVS_SIZE (1200), VCR_SIFT_SIZE (1920), VCR_MIN_SHOT (12).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ENV = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
WS = Path("/content/vcr")
OUT = WS / "out"
CONDA = Path("/content/miniforge3/bin/colmap")
MVS_SIZE = os.environ.get("VCR_MVS_SIZE", "1200")
SIFT_SIZE = os.environ.get("VCR_SIFT_SIZE", "1920")
MIN_SHOT = int(os.environ.get("VCR_MIN_SHOT", "12"))


def run(cmd, env=ENV):
    t = time.time()
    print(f"\n=== {' '.join(str(c) for c in cmd)}", flush=True)
    log = WS / "last_cmd.log"
    with open(log, "wb") as lf:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=lf, env=env)
        while proc.poll() is None:
            time.sleep(30)
            try:
                lines = log.read_text(errors="replace").splitlines()
                if lines:
                    print("   |", lines[-1][:150], flush=True)
            except OSError:
                pass
        if proc.returncode:
            try:
                tail = log.read_text(errors="replace").splitlines()[-20:]
                print("--- cmd failed ---\n" + "\n".join(tail), flush=True)
            except OSError:
                pass
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"=== done in {time.time() - t:.1f}s", flush=True)


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                        "--format=csv,noheader"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("no GPU", flush=True)
    if not CONDA.exists():
        if not Path("/content/miniforge3/bin/conda").exists():
            run(["curl", "-sL", "-o", "/content/miniforge.sh",
                 "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"])
            run(["bash", "/content/miniforge.sh", "-b", "-p", "/content/miniforge3"])
        run(["/content/miniforge3/bin/conda", "install", "-y", "-c", "conda-forge",
             "colmap", "cuda-version=12.6"])
    if not (WS / "images").exists():
        run(["unzip", "-q", "-o", str(WS / "frames.zip"), "-d", str(WS)])
        ex = WS / "frames"
        if ex.exists():
            ex.rename(WS / "images")


def reconstruct_shot(shot):
    sid = shot["id"]
    files = shot["files"]
    tag = f"shot{sid}"
    img = WS / f"images_{tag}"
    db = WS / f"db_{tag}.db"
    sparse = WS / f"sparse_{tag}"
    dense = WS / f"dense_{tag}"
    fused = OUT / f"fused_{tag}.ply"
    if fused.exists():
        print(f"[{tag}] fused exists, skip", flush=True)
        return {"shot": sid, "status": "cached", "n_frames": len(files)}
    img.mkdir(exist_ok=True)
    for fn in files:
        dst = img / fn
        if not dst.exists():
            shutil.copyfile(WS / "images" / fn, dst)
    masks = WS / f"masks_{tag}"
    masks.mkdir(exist_ok=True)
    if (WS / "watermark_mask.png").exists():
        for fn in files:
            dst = masks / (Path(fn).stem + ".png")
            if not dst.exists():
                shutil.copyfile(WS / "watermark_mask.png", dst)
    try:
        if not (WS / f".feat_{tag}").exists():
            if db.exists():
                db.unlink()
            run([CONDA, "feature_extractor", "--database_path", db, "--image_path", img,
                 "--ImageReader.mask_path", masks, "--ImageReader.single_camera", "1",
                 "--FeatureExtraction.use_gpu", "1",
                 "--SiftExtraction.max_image_size", SIFT_SIZE,
                 "--SiftExtraction.max_num_features", "16384"])
            (WS / f".feat_{tag}").touch()
        if not (WS / f".match_{tag}").exists():
            run([CONDA, "sequential_matcher", "--database_path", db,
                 "--FeatureMatching.use_gpu", "1",
                 "--SequentialMatching.overlap", "10",
                 "--SequentialMatching.quadratic_overlap", "1"])
            (WS / f".match_{tag}").touch()
        sparse.mkdir(parents=True, exist_ok=True)
        if not (WS / f".map_{tag}").exists():
            run([CONDA, "mapper", "--database_path", db, "--image_path", img,
                 "--output_path", sparse, "--Mapper.multiple_models", "0",
                 "--Mapper.ba_global_max_num_iterations", "25"])
            (WS / f".map_{tag}").touch()
        models = sorted([m for m in sparse.iterdir() if m.is_dir() and (m / "images.bin").exists()])
        if not models:
            return {"shot": sid, "status": "no_model", "n_frames": len(files)}
        best = max(models, key=lambda m: (m / "images.bin").stat().st_size)
        dense.mkdir(parents=True, exist_ok=True)
        if not (dense / "stereo" / "patch-match.cfg").exists():
            run([CONDA, "image_undistorter", "--image_path", img, "--input_path", best,
                 "--output_path", dense, "--output_type", "COLMAP",
                 "--max_image_size", MVS_SIZE])
        depth_dir = dense / "stereo" / "depth_maps"
        n_depth = len(list(depth_dir.glob("*.geometric.bin"))) if depth_dir.exists() else 0
        if n_depth < len(files):
            run([CONDA, "patch_match_stereo", "--workspace_path", dense,
                 "--PatchMatchStereo.max_image_size", MVS_SIZE,
                 "--PatchMatchStereo.geom_consistency", "true",
                 "--PatchMatchStereo.window_radius", "5",
                 "--PatchMatchStereo.num_samples", "15",
                 "--PatchMatchStereo.num_iterations", "5"])
        if not fused.exists():
            run([CONDA, "stereo_fusion", "--workspace_path", dense, "--input_type", "geometric",
                 "--StereoFusion.min_num_pixels", "3", "--StereoFusion.max_reproj_error", "2.0",
                 "--output_path", fused])
        txt = WS / f"txt_{tag}"
        run([CONDA, "model_converter", "--input_path", best, "--output_path", txt,
             "--output_type", "TXT"])
        nreg = sum(1 for ln in (txt / "images.txt").read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")) // 2
        return {"shot": sid, "status": "ok", "n_frames": len(files),
                "n_registered": nreg, "n_models": len(models),
                "fused_bytes": fused.stat().st_size}
    except subprocess.CalledProcessError as e:
        print(f"[{tag}] FAILED rc={e.returncode}, skipping", flush=True)
        return {"shot": sid, "status": "failed", "n_frames": len(files)}


def main():
    setup()
    shots = json.loads((WS / "shots.json").read_text())["shots"]
    shots = [s for s in shots if s["end"] - s["start"] >= MIN_SHOT]
    print(f"processing {len(shots)} shots (>= {MIN_SHOT} frames)", flush=True)
    results = []
    for shot in shots:
        print(f"\n########## SHOT {shot['id']} ({shot['end']-shot['start']} frames) ##########", flush=True)
        results.append(reconstruct_shot(shot))
    (OUT / "shots_stats.json").write_text(json.dumps(results, indent=1))
    print("\n===== SHOTS STATS =====")
    print(json.dumps(results, indent=1), flush=True)
    print("MULTISHOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()