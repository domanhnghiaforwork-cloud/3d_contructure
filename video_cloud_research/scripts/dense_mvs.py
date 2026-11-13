"""MVS dense cho cloud dựng lại: colmap CUDA (conda-forge), chạy theo từng model.

Bản apt/pycolmap không có CUDA nên PatchMatch phải dùng bản conda-forge.
Resume theo marker; heartbeat in dòng log cuối mỗi 30 s.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

WS = Path("/content/vcr")
ENV = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
CONDA_COLMAP = Path("/content/miniforge3/bin/colmap")


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
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"=== done in {time.time() - t:.1f}s", flush=True)


def main():
    import pycolmap
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["merged02"])
    args = ap.parse_args()

    def stage(s):
        print(f"\n########## {s} ##########", flush=True)

    stage("conda colmap (CUDA)")
    if not CONDA_COLMAP.exists():
        if not Path("/content/miniforge3/bin/conda").exists():
            run(["curl", "-sL", "-o", "/content/miniforge.sh",
                 "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"])
            run(["bash", "/content/miniforge.sh", "-b", "-p", "/content/miniforge3"])
        run(["/content/miniforge3/bin/conda", "install", "-y", "-c", "conda-forge",
             "colmap", "cuda-version=12.6"])

    out = WS / "out"
    out.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        sparse = WS / "sparse" / model
        if not (sparse / "images.bin").exists():
            raise RuntimeError(f"{model} missing")
        rec = pycolmap.Reconstruction(str(sparse))
        print(f"\n===== MODEL {model}: {rec.summary()}", flush=True)

        dense = WS / f"dense_{model}"
        stage(f"image_undistorter {model}")
        if not (dense / "stereo" / "patch-match.cfg").exists():
            run(["colmap", "image_undistorter",
                 "--image_path", WS / "images", "--input_path", sparse,
                 "--output_path", dense, "--output_type", "COLMAP",
                 "--max_image_size", os.environ.get("VCR_MVS_SIZE","1200")])

        stage(f"patch_match_stereo {model} (CUDA)")
        depth_dir = dense / "stereo" / "depth_maps"
        n_img = rec.num_reg_images()
        n_depth = len(list(depth_dir.glob("*.geometric.bin"))) if depth_dir.exists() else 0
        if n_depth < n_img:
            run([CONDA_COLMAP, "patch_match_stereo",
                 "--workspace_path", dense,
                 "--PatchMatchStereo.max_image_size", os.environ.get("VCR_MVS_SIZE","1200"),
                 "--PatchMatchStereo.geom_consistency", "true",
                 "--PatchMatchStereo.window_radius", "5",
                 "--PatchMatchStereo.num_samples", "15",
                 "--PatchMatchStereo.num_iterations", "5"])

        stage(f"stereo_fusion {model}")
        fused = out / f"fused_{model}.ply"
        if not fused.exists():
            run([CONDA_COLMAP, "stereo_fusion",
                 "--workspace_path", dense, "--input_type", "geometric",
                 "--StereoFusion.min_num_pixels", "3",
                 "--StereoFusion.max_reproj_error", "2.0",
                 "--output_path", fused])

        stage(f"stats {model}")
        import numpy as np
        from trimesh.exchange.ply import load_ply
        with open(fused, "rb") as h:
            data = load_ply(h)
        pts = np.asarray(data["vertices"], dtype=float)
        stats = {"model": model, "n_points": int(len(pts)),
                 "bbox_min": pts.min(axis=0).tolist(),
                 "bbox_max": pts.max(axis=0).tolist()}
        (out / f"stats_{model}.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1), flush=True)
    print("DENSE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
