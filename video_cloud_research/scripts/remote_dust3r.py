"""Dựng point cloud từ frame video bằng DUSt3R (GPU, không cần SfM cổ điển).

Cài dust3r + weights, scene graph sliding-window, global alignment,
xuất point cloud màu + confidence lọc. Chạy trên Colab T4/A100.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/content/vcr")


def run(cmd):
    t = time.time()
    print(f"\n=== {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)
    print(f"=== done in {time.time() - t:.1f}s", flush=True)


def main():
    t0 = time.time()
    if not Path("/content/dust3r").exists():
        run(["git", "clone", "--depth", "1", "https://github.com/naver/dust3r", "/content/dust3r"])
    run([sys.executable, "-m", "pip", "install", "-q",
         "einops", "trimesh", "huggingface_hub", "safetensors", "scipy",
         "roma", "opencv-python-headless"])
    if not Path("/content/dust3r/croco/models").exists():
        run(["git", "-C", "/content/dust3r", "submodule", "update", "--init", "--recursive", "--depth", "1"])
    sys.path.insert(0, "/content/dust3r")
    import numpy as np
    import torch
    # checkpoint chính thức Naver chứa argparse namespace; torch>=2.6 cần weights_only=False
    if not getattr(torch.load, "_dust3r_patched", False):
        _orig_load = torch.load
        def _patched_load(*a, **k):
            return _orig_load(*a, **{**k, "weights_only": False})
        _patched_load._dust3r_patched = True
        torch.load = _patched_load
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)

    from huggingface_hub import hf_hub_download
    repo = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(repo).to(device)
    except TypeError:
        import json as _json
        from safetensors.torch import load_file
        cfg = _json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        if isinstance(cfg.get("img_size"), int):
            cfg["img_size"] = [cfg["img_size"], cfg["img_size"]]
        model = AsymmetricCroCo3DStereo(**cfg)
        model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
        model = model.to(device)

    if not (WS / "images").exists():
        zip_name = os.environ.get("VCR_ZIP", "frames_4fps.zip")
        run(["unzip", "-q", "-o", str(WS / zip_name), "-d", str(WS)])
        (WS / zip_name.replace(".zip", "")).rename(WS / "images")
    paths = sorted(str(p) for p in (WS / "images").glob("*.jpg"))[::int(os.environ.get("VCR_STRIDE", "2"))]
    print("frames:", len(paths), flush=True)

    images = load_images(paths, size=384, verbose=True)
    pairs = make_pairs(images, scene_graph=os.environ.get("VCR_GRAPH","logwin"), prefilter=None, symmetrize=True)
    print("pairs:", len(pairs), flush=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=int(os.environ.get("VCR_NITER","1000")), lr=0.01, schedule="cosine")
    print("global alignment loss:", float(loss), flush=True)

    pts3d = scene.get_pts3d()
    conf = scene.get_conf_masks() if hasattr(scene, "get_conf_masks") else None
    imgs = scene.imgs

    def to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    cache = WS / "out" / "scene_cache.npz"
    if os.environ.get("VCR_RECOMPUTE") == "1" and cache.exists():
        cache.unlink()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        P, C = z["points"], z["colors"]
    else:
        all_p, all_c = [], []
        for i in range(len(pts3d)):
            p = to_np(pts3d[i]).reshape(-1, 3)
            im = to_np(imgs[i])
            if im.ndim == 3 and im.shape[0] == 3:
                im = im.transpose(1, 2, 0)
            col = (np.clip(im, 0, 1) * 255).astype(np.uint8).reshape(-1, 3)
            if conf is not None:
                m = to_np(conf[i]).reshape(-1) > float(os.environ.get("VCR_CONF", "1.5"))
                p, col = p[m], col[m]
            all_p.append(p)
            all_c.append(col)
        P = np.concatenate(all_p)
        C = np.concatenate(all_c)
        np.savez_compressed(cache, points=P, colors=C)
    print("raw points:", len(P), flush=True)

    import trimesh
    out = WS / "out"
    out.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=P, vertex_colors=C, process=False).export(out / "fused_dust3r.ply")
    stats = {"n_points": int(len(P)), "bbox_min": P.min(0).tolist(),
             "bbox_max": P.max(0).tolist(), "loss": float(loss),
             "elapsed_s": round(time.time() - t0, 1)}
    (out / "stats_dust3r.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)
    print("DUST3R COMPLETE", flush=True)


if __name__ == "__main__":
    main()