"""Căn chỉnh từng shot vào hệ tham chiếu rồi HỢP NHẤT cloud đã căn chỉnh (đa góc).

Mỗi shot dựng riêng (remote_colmap_shots) -> align_and_compare từng shot (scale-search)
-> ghép các recon_aligned.ply (cùng hệ ref, đúng scale) -> cloud đa góc phủ rộng hơn.
Hợp nhất SAU căn chỉnh tránh lỗi lệch scale của merge model thô (bài học lần cũ).

Xuất alignment_ref/recon_aligned.ply + recon_attributes.npz (points/colors) để
estimate_normals + evaluate_raamac chạy tiếp như cloud đơn.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
PY = os.environ.get("PYTHON") or sys.executable
ALIGN = HERE / "align_and_compare.py"
REF_ATTRS = PKG / "data" / "cloud_attributes.npz"
REF_PLY = PKG / "data" / "raw" / "week26.ply"


def run(cmd):
    print("\n+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--voxel", type=float, default=0.05)
    args = ap.parse_args()
    out = args.out_dir or (HERE.parent / "artifacts" / f"week{args.week}")
    shot_clouds = sorted(out.glob("fused_shot*.ply"))
    if not shot_clouds:
        raise SystemExit("Không có fused_shot*.ply — chạy run_colmap_shots.py trước")

    per_shot = []
    all_pts, all_col = [], []
    for sc in shot_clouds:
        sid = sc.stem.replace("fused_shot", "")
        adir = out / f"alignment_shot{sid}"
        run([PY, ALIGN, "--recon", sc, "--ref-attributes", REF_ATTRS,
             "--out-dir", adir, "--scale-search"])
        m = json.loads((adir / "metrics.json").read_text())
        per_shot.append({"shot": sid, "scale": round(m["scale"], 3),
                         "chamfer_m": round(m["chamfer_mean_m"], 3),
                         "median_m": round(m["recon_to_ref_m"]["median"], 3),
                         "coverage_ref_0.2": round(m["coverage_ref_at_tau"]["0.2"], 3),
                         "n_recon": m["n_recon"]})
        mesh = trimesh.load(str(adir / "recon_aligned.ply"), process=False)
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        try:
            col = np.asarray(mesh.visual.vertex_colors)[:, :3]
            if len(col) != len(pts):
                col = np.full((len(pts), 3), 128, np.uint8)
        except Exception:
            col = np.full((len(pts), 3), 128, np.uint8)
        all_pts.append(pts)
        all_col.append(col)
        print(f"[shot {sid}] aligned: scale={m['scale']:.3f} "
              f"chamfer={m['chamfer_mean_m']:.3f} cov={m['coverage_ref_at_tau']['0.2']:.3f}", flush=True)

    pts = np.concatenate(all_pts)
    col = np.concatenate(all_col)
    # voxel downsample khử trùng lặp giữa các shot
    keys = np.floor(pts / args.voxel).astype(np.int64)
    kv = np.ascontiguousarray(keys).view(np.dtype((np.void, keys.dtype.itemsize * 3)))
    _, first = np.unique(kv, return_index=True)
    pts, col = pts[first], col[first]

    aref = out / "alignment_ref"
    aref.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=pts, vertex_colors=col.astype(np.uint8),
                    process=False).export(aref / "recon_aligned.ply")
    np.savez_compressed(aref / "recon_attributes.npz",
                        points=pts, normals=np.zeros_like(pts),
                        colors=col.astype(np.float64) / 255.0)

    from scipy.spatial import cKDTree
    ref = np.asarray(np.load(REF_ATTRS)["points"], dtype=float)
    tree_m = cKDTree(pts)
    tree_r = cKDTree(ref)
    d_r2f, _ = tree_r.query(pts, workers=4)
    d_f2r, _ = tree_m.query(ref, workers=4)
    metrics = {
        "n_recon": int(len(pts)), "n_ref": int(len(ref)), "scale": None,
        "recon_to_ref_m": {"mean": float(d_r2f.mean()), "median": float(np.median(d_r2f))},
        "ref_to_recon_m": {"mean": float(d_f2r.mean()), "median": float(np.median(d_f2r))},
        "chamfer_mean_m": float(0.5 * (d_r2f.mean() + d_f2r.mean())),
        "coverage_ref_at_tau": {str(t): float(np.mean(d_f2r <= t)) for t in (0.05, 0.1, 0.2, 0.3)},
        "coverage_recon_at_tau": {str(t): float(np.mean(d_r2f <= t)) for t in (0.05, 0.1, 0.2, 0.3)},
    }
    (aref / "metrics.json").write_text(json.dumps(metrics, indent=1))
    summary = {"week": args.week, "n_shots_merged": len(shot_clouds),
               "n_points_merged": int(len(pts)), "voxel": args.voxel,
               "merged_coverage_ref_0.2": metrics["coverage_ref_at_tau"]["0.2"],
               "merged_chamfer_m": metrics["chamfer_mean_m"],
               "per_shot": per_shot}
    (out / "merge_summary.json").write_text(json.dumps(summary, indent=1))
    final = out / f"week{args.week}_final.ply"
    shutil.copyfile(aref / "recon_aligned.ply", final)
    run([PY, HERE / "render_cloud.py", "--cloud", final, "--ref", REF_PLY,
         "--out", out / "renders_final", "--title", f"week{args.week} FINAL (đa shot)"])
    print(f"\nFINAL DELIVERABLE -> {final} ({final.stat().st_size/1e6:.1f} MB)", flush=True)
    print("\n===== MERGE SUMMARY =====")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()