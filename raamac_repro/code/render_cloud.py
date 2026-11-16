"""Render point cloud ra PNG để TỰ KIỂM TRA hình học (không cần browser).

Vẽ 4 khung: 3D phối cảnh + chiếu bằng XY + chiếu đứng XZ + chiếu cạnh YZ.
Nếu có --ref (cloud chuyên nghiệp), chồng lên màu xám để đối chiếu căn chỉnh.
Downsample để render nhanh. Dùng matplotlib Agg (headless).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def load(path, max_points, seed=7):
    m = trimesh.load(str(path), process=False)
    pts = np.asarray(m.vertices, dtype=float)
    try:
        col = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
        if len(col) != len(pts):
            col = None
    except Exception:
        col = None
    if len(pts) > max_points:
        idx = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        col = col[idx] if col is not None else None
    return pts, col


def color_by_height(pts):
    z = pts[:, 2]
    zn = (z - z.min()) / max(np.ptp(z), 1e-9)
    return plt.cm.viridis(zn)[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-points", type=int, default=40000)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pts, col = load(args.cloud, args.max_points)
    if col is None:
        col = color_by_height(pts)
    ref = None
    if args.ref:
        ref, _ = load(args.ref, args.max_points)

    views = [("persp", None), ("top_XY", (0, 1)), ("front_XZ", (0, 2)), ("side_YZ", (1, 2))]
    for name, axes2d in views:
        fig = plt.figure(figsize=(8, 8), dpi=110)
        if axes2d is None:
            ax = fig.add_subplot(111, projection="3d")
            if ref is not None:
                ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=0.3, c="#bbbbbb",
                           alpha=0.25, depthshade=False)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.6, c=col,
                       alpha=0.9, depthshade=False)
            ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
            ax.view_init(elev=22, azim=-55)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        else:
            ax = fig.add_subplot(111)
            a, b = axes2d
            if ref is not None:
                ax.scatter(ref[:, a], ref[:, b], s=0.4, c="#bbbbbb", alpha=0.3)
            ax.scatter(pts[:, a], pts[:, b], s=0.6, c=col, alpha=0.9)
            ax.set_aspect("equal")
            ax.set_xlabel("XYZ"[a]); ax.set_ylabel("XYZ"[b])
            ax.grid(alpha=0.2)
        title = args.title or args.cloud.stem
        ax.set_title(f"{title} — {name}  (n={len(pts):,}"
                     + (f", ref xám={len(ref):,})" if ref is not None else ")"))
        fig.tight_layout()
        out = args.out / f"{name}.png"
        fig.savefig(out, facecolor="#0e1116")
        plt.close(fig)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()