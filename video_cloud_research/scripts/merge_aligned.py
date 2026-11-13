"""Hợp nhất các cloud dựng lại ĐÃ căn chỉnh vào hệ ref thành một bộ đầu vào đánh giá."""
import argparse
from pathlib import Path

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pts, nms, cols = [], [], []
    for d in args.dirs:
        with np.load(Path(d) / "recon_attributes.npz", allow_pickle=False) as a:
            pts.append(a["points"])
            nms.append(a["normals"])
            cols.append(a["colors"])
    p = np.concatenate(pts)
    n = np.concatenate(nms)
    c = np.concatenate(cols)
    np.savez_compressed(args.out_dir / "recon_attributes.npz", points=p, normals=n, colors=c)
    mesh = trimesh.Trimesh(vertices=p, vertex_normals=n,
                           vertex_colors=(c * 255).astype(np.uint8), process=False)
    mesh.export(args.out_dir / "recon_aligned.ply")
    print(f"union: {len(p)} points from {len(args.dirs)} clouds")


if __name__ == "__main__":
    main()