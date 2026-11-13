"""Ước lượng pháp tuyến cho cloud dựng lại (DUSt3R/MASt3R không có normals).

PCA cục bộ k láng giềng, vector riêng ứng trị nhỏ nhất = pháp tuyến. Plane mode
của pipeline chính dùng abs(dot) nên dấu pháp tuyến không quan trọng. Giữ nguyên
points (khung ref) để progress_baseline áp G3 ref->IFC chính xác.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(pts, k=16):
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1, workers=4)
    nb = pts[idx[:, 1:]]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]
    n = n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    a = np.load(args.aligned_dir / "recon_attributes.npz")
    pts = np.asarray(a["points"], dtype=float)
    colors = np.asarray(a["colors"], dtype=float)
    n = estimate_normals(pts, args.k)
    np.savez_compressed(args.out_dir / "recon_attributes.npz",
                        points=pts, normals=n, colors=colors)
    import trimesh
    trimesh.Trimesh(vertices=pts, vertex_normals=n,
                    vertex_colors=(colors * 255).astype(np.uint8),
                    process=False).export(args.out_dir / "recon_aligned.ply")
    print(f"normals estimated for {len(pts)} points (k={args.k}) -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()