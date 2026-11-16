"""COLMAP SfM+MVS tái hiện RAAMAC trên Colab T4 (CUDA toàn trình).

Cải tiến so với lần dựng cũ (113 frame @2fps, vỡ 3 model, merge lệch scale):
- Frame dày 5fps full 1080p (268+ ảnh) -> overlap lớn -> MỘT model nhất quán.
- SIFT full 1920px, 16384 feature, GPU.
- Sequential matching overlap 10 + quadratic (khép vòng quỹ đạo).
- Mapper KHÔNG merge thủ công (tránh lệch scale); chọn model lớn nhất, báo cáo
  số model + tỉ lệ ảnh đăng ký để tự kiểm tra tính nhất quán.
- PatchMatch CUDA full-res, geom_consistency, fusion lọc nhiễu.

Resume theo marker; heartbeat in dòng log cuối mỗi 30s. Xuất fused.ply + stats.json.
Env: VCR_MVS_SIZE (default 1600), VCR_SIFT_SIZE (default 1920), VCR_OVERLAP (10).
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
CONDA = Path("/content/miniforge3/bin/colmap")
MVS_SIZE = os.environ.get("VCR_MVS_SIZE", "1200")
SIFT_SIZE = os.environ.get("VCR_SIFT_SIZE", "1920")
OVERLAP = os.environ.get("VCR_OVERLAP", "10")


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
                    print("   |", lines[-1][:160], flush=True)
            except OSError:
                pass
        if proc.returncode:
            try:
                tail = log.read_text(errors="replace").splitlines()[-30:]
                print("--- cmd failed, log tail ---\n" + "\n".join(tail), flush=True)
            except OSError:
                pass
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"=== done in {time.time() - t:.1f}s", flush=True)


def stage(s):
    print(f"\n########## {s} ##########", flush=True)


def main():
    WS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    stage("setup conda colmap (CUDA)")
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
    run([CONDA, "-h"])

    if not IMAGES.exists():
        zip_name = os.environ.get("VCR_ZIP", "frames.zip")
        run(["unzip", "-q", "-o", str(WS / zip_name), "-d", str(WS)])
        extracted = WS / zip_name.replace(".zip", "")
        if extracted.exists():
            extracted.rename(IMAGES)
    n_img = len(list(IMAGES.glob("*.jpg")))
    print(f"images: {n_img}", flush=True)

    stage("prepare per-image watermark masks")
    MASKS = WS / "masks"
    MASKS.mkdir(exist_ok=True)
    mask_src = WS / "watermark_mask.png"
    if mask_src.exists():
        for img in IMAGES.glob("*.jpg"):
            dst = MASKS / (img.stem + ".png")
            if not dst.exists():
                shutil.copyfile(mask_src, dst)
        print(f"masks: {len(list(MASKS.glob('*.png')))}", flush=True)

    stage("feature_extractor (GPU SIFT full-res)")
    if not (WS / ".features_done").exists():
        if DB.exists():
            DB.unlink()
        run([CONDA, "feature_extractor",
             "--database_path", DB, "--image_path", IMAGES,
             "--ImageReader.mask_path", MASKS,
             "--ImageReader.single_camera", "1",
             "--FeatureExtraction.use_gpu", "1",
             "--SiftExtraction.max_image_size", SIFT_SIZE,
             "--SiftExtraction.max_num_features", "16384"])
        (WS / ".features_done").touch()

    stage("exhaustive_matcher (GPU)")
    if not (WS / ".matched").exists():
        run([CONDA, "exhaustive_matcher",
             "--database_path", DB, "--FeatureMatching.use_gpu", "1"])
        (WS / ".matched").touch()

    stage("mapper")
    SPARSE.mkdir(parents=True, exist_ok=True)
    if not (WS / ".mapped").exists():
        run([CONDA, "mapper",
             "--database_path", DB, "--image_path", IMAGES,
             "--output_path", SPARSE,
             "--Mapper.multiple_models", "0",
             "--Mapper.ba_global_max_num_iterations", "50",
             "--Mapper.ba_local_max_num_iterations", "25"])
        (WS / ".mapped").touch()
    models = sorted([m for m in SPARSE.iterdir() if m.is_dir()])
    print("sparse models:", [m.name for m in models], flush=True)

    def n_registered(m):
        return (m / "images.bin").stat().st_size

    best = max(models, key=n_registered)
    (OUT / "best_model.txt").write_text(best.name)

    stage("image_undistorter")
    DENSE.mkdir(parents=True, exist_ok=True)
    if not (DENSE / "stereo" / "patch-match.cfg").exists():
        run([CONDA, "image_undistorter",
             "--image_path", IMAGES, "--input_path", best,
             "--output_path", DENSE, "--output_type", "COLMAP",
             "--max_image_size", MVS_SIZE])

    stage("patch_match_stereo (CUDA)")
    depth_dir = DENSE / "stereo" / "depth_maps"
    n_depth = len(list(depth_dir.glob("*.geometric.bin"))) if depth_dir.exists() else 0
    if n_depth < n_img:
        run([CONDA, "patch_match_stereo",
             "--workspace_path", DENSE,
             "--PatchMatchStereo.max_image_size", MVS_SIZE,
             "--PatchMatchStereo.geom_consistency", "true",
             "--PatchMatchStereo.window_radius", "5",
             "--PatchMatchStereo.num_samples", "15",
             "--PatchMatchStereo.num_iterations", "5"])

    stage("stereo_fusion")
    if not (OUT / "fused.ply").exists():
        run([CONDA, "stereo_fusion",
             "--workspace_path", DENSE, "--input_type", "geometric",
             "--StereoFusion.min_num_pixels", "3",
             "--StereoFusion.max_reproj_error", "2.0",
             "--output_path", OUT / "fused.ply"])

    stage("stats")
    import sys
    import numpy as np
    try:
        from trimesh.exchange.ply import load_ply
    except ModuleNotFoundError:
        run([sys.executable, "-m", "pip", "install", "-q", "trimesh"])
        from trimesh.exchange.ply import load_ply
    txt = WS / "model_txt"
    run([CONDA, "model_converter",
         "--input_path", best, "--output_path", txt, "--output_type", "TXT"])
    model_sizes = {}
    for m in models:
        t = WS / f"mtxt_{m.name}"
        run([CONDA, "model_converter", "--input_path", m,
             "--output_path", t, "--output_type", "TXT"])
        nreg = sum(1 for ln in (t / "images.txt").read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")) // 2
        model_sizes[m.name] = nreg
    n_registered_best = model_sizes[best.name]
    with open(OUT / "fused.ply", "rb") as h:
        data = load_ply(h)
    pts = np.asarray(data["vertices"], dtype=float)
    stats = {
        "n_images": n_img,
        "n_models": len(models),
        "model_registrations": model_sizes,
        "best_model": best.name,
        "n_registered": n_registered_best,
        "registration_ratio": round(n_registered_best / n_img, 3),
        "n_points": int(len(pts)),
        "bbox_min": pts.min(axis=0).tolist(),
        "bbox_max": pts.max(axis=0).tolist(),
        "mvs_size": int(MVS_SIZE), "sift_size": int(SIFT_SIZE),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)
    print("RECONSTRUCTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()