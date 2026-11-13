"""Dựng point cloud từ frame drone bằng COLMAP (SfM + MVS) trên Colab T4.

Chạy qua: colab --auth=oauth2 exec -s <session> -f colmap_reconstruct.py
Script tự cài COLMAP, chạy từng giai đoạn (resume được), xuất fused.ply.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ENV = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}

WS = Path("/content/vcr")
IMAGES = WS / "images"
OUT = WS / "out"
DB = WS / "database.db"
SPARSE = WS / "sparse"
DENSE = WS / "dense"


def run(cmd, **kw):
    t = time.time()
    print(f"\n=== {' '.join(str(c) for c in cmd)}", flush=True)
    log = WS / "last_cmd.log"
    with open(log, "wb") as lf:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=lf, env=ENV)
        while proc.poll() is None:
            time.sleep(30)
            try:
                lines = log.read_text(errors="replace").splitlines()
                if lines:
                    print("   |", lines[-1][:160], flush=True)
            except OSError:
                pass
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"=== done in {time.time() - t:.1f}s", flush=True)


def run_with_cpu_fallback(cmd):
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        print("GPU path failed, retrying with CPU", flush=True)
        cpu_cmd = ["0" if c == "1" else c for c in cmd]
        run(cpu_cmd)


def stage(name):
    print(f"\n########## STAGE {name} ##########", flush=True)


def main():
    WS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    stage("setup")
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                        "--format=csv,noheader"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("no GPU (CPU session)", flush=True)
    if not shutil.which("colmap"):
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "colmap"])
    run(["colmap", "-h"], stdout=subprocess.DEVNULL)

    if not IMAGES.exists():
        zip_name = os.environ.get("VCR_ZIP", "frames_4fps.zip")
        run(["unzip", "-q", "-o", str(WS / zip_name), "-d", str(WS)])
        extracted = WS / zip_name.replace(".zip", "")
        extracted.rename(IMAGES)
    n_img = len(list(IMAGES.glob("*.jpg")))
    print(f"images: {n_img}", flush=True)

    stage("feature_extractor")
    if not (WS / ".features_done").exists():
        if DB.exists():
            DB.unlink()
        run_with_cpu_fallback(["colmap", "feature_extractor",
             "--database_path", DB, "--image_path", IMAGES,
             "--ImageReader.mask_path", WS / "watermark_mask.png",
             "--SiftExtraction.use_gpu", "0",
             "--SiftExtraction.max_image_size", "1600",
             "--SiftExtraction.max_num_features", "8192"])
        (WS / ".features_done").touch()

    stage("sequential_matcher")
    if not (WS / ".matched").exists():
        run_with_cpu_fallback(["colmap", "sequential_matcher",
             "--database_path", DB, "--SiftMatching.use_gpu", "0",
             "--SequentialMatching.overlap", "15",
             "--SequentialMatching.quadratic_overlap", "1"])
        (WS / ".matched").touch()

    stage("mapper")
    SPARSE.mkdir(parents=True, exist_ok=True)
    if not (SPARSE / "0").exists():
        run(["colmap", "mapper",
             "--database_path", DB, "--image_path", IMAGES,
             "--output_path", SPARSE,
             "--Mapper.ba_global_max_num_iterations", "25"])
    models = sorted(SPARSE.iterdir())
    print("sparse models:", [m.name for m in models], flush=True)
    best = max(models, key=lambda m: (m / "images.bin").stat().st_size)
    (OUT / "best_model.txt").write_text(best.name)
    run(["colmap", "model_converter",
         "--input_path", best, "--output_path", OUT / "sparse.ply",
         "--output_type", "PLY"])

    stage("image_undistorter")
    DENSE.mkdir(parents=True, exist_ok=True)
    if os.environ.get("VCR_SFM_ONLY") == "1":
        print("SFM_ONLY: stopping before MVS", flush=True)
        return
    if not (DENSE / "stereo" / "patch-match.cfg").exists():
        run(["colmap", "image_undistorter",
             "--image_path", IMAGES, "--input_path", best,
             "--output_path", DENSE, "--output_type", "COLMAP",
             "--max_image_size", "1200"])

    stage("patch_match_stereo")
    depth_dir = DENSE / "stereo" / "depth_maps"
    n_depth = len(list(depth_dir.glob("*.geometric.bin"))) if depth_dir.exists() else 0
    if n_depth < n_img:
        run(["colmap", "patch_match_stereo",
             "--workspace_path", DENSE,
             "--PatchMatchStereo.max_image_size", "1200",
             "--PatchMatchStereo.geom_consistency", "true",
             "--PatchMatchStereo.window_radius", "5",
             "--PatchMatchStereo.num_samples", "15",
             "--PatchMatchStereo.num_iterations", "5"])

    stage("stereo_fusion")
    if not (OUT / "fused.ply").exists():
        run(["colmap", "stereo_fusion",
             "--workspace_path", DENSE,
             "--input_type", "geometric",
             "--min_num_pixels", "3",
             "--max_reproj_error", "2.0",
             "--output_path", OUT / "fused.ply"])

    stage("stats")
    import numpy as np
    import trimesh
    txt = WS / "model_txt"
    run(["colmap", "model_converter",
         "--input_path", best, "--output_path", txt, "--output_type", "TXT"])
    n_registered = sum(
        1 for line in (txt / "images.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ) // 2
    fused = trimesh.load(OUT / "fused.ply", process=False)
    pts = np.asarray(fused.vertices)
    stats = {
        "n_images": n_img,
        "n_registered": n_registered,
        "n_points": int(len(pts)),
        "bbox_min": pts.min(axis=0).tolist(),
        "bbox_max": pts.max(axis=0).tolist(),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)
    print("RECONSTRUCTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()