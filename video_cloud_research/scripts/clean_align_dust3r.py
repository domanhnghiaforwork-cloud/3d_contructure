"""Khử nhiễu cloud DUSt3R + căn vào mỏ neo chuyên nghiệp (week26.ply) đúng hướng.

Pipeline: lọc vegetation bằng màu (bỏ cây xanh) -> căn thô (PCA + 24 phép quay +
tìm scale) vào khối nhà chuyên nghiệp khung ref -> crop về bbox khối nhà + lọc
outlier (mét) -> ICP similarity mịn -> xuất cloud SẠCH khung ref (cho eval đóng
băng, progress_baseline tự áp G3) và khung IFC (áp G3, cho viewer) + metrics.

Mỏ neo = cloud chuyên nghiệp đã đăng ký tốt (G3). Không tune nhãn/test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from align_and_compare import pca_frame, signed_permutations, umeyama, apply_sim
from refine_registration import pca_normals, apply_t

ROOT = Path(__file__).resolve().parents[2]


def dense_core(pts, q=0.5, k=8):
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1, workers=4)
    md = d[:, 1:].mean(axis=1)
    return pts[md <= np.quantile(md, q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, required=True, help="fused.ply DUSt3R")
    ap.add_argument("--ref-attributes", type=Path,
                    default=ROOT / "artifacts/week26_raw_diagnostics/cloud_attributes.npz")
    ap.add_argument("--reg-inputs", type=Path,
                    default=ROOT / "artifacts/week26_registration_v3/registration_inputs.npz")
    ap.add_argument("--g3", type=Path,
                    default=ROOT / "artifacts/week26_registration_v3/candidate_transform.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-points", type=int, default=1_200_000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    from trimesh.exchange.ply import load_ply
    with open(args.recon, "rb") as h:
        data = load_ply(h)
    pts = np.asarray(data["vertices"], float)
    raw_c = data.get("vertex_colors")
    if raw_c is not None:
        col = np.asarray(raw_c, np.uint8)[:, :3]
    elif data.get("red") is not None:
        col = np.stack([data[k] for k in ("red", "green", "blue")], 1).astype(np.uint8)
    else:
        col = np.full((len(pts), 3), 128, np.uint8)
    if len(pts) > args.max_points:
        sel = rng.choice(len(pts), args.max_points, replace=False)
        pts, col = pts[sel], col[sel]
    n0 = len(pts)

    # 1) lọc vegetation: bỏ điểm xanh lá trội (cây), giữ nhà trắng/xám
    r, g, b = col[:, 0].astype(int), col[:, 1].astype(int), col[:, 2].astype(int)
    veg = (g > r + 8) & (g > b + 8)
    pts, col = pts[~veg], col[~veg]
    n1 = len(pts)
    print(f"vegetation filter: {n0} -> {n1} (bo {n0-n1} diem xanh)", flush=True)

    # mỏ neo: khối nhà chuyên nghiệp khung ref (crop theo bbox IFC nghịch đảo G3)
    with np.load(args.ref_attributes, allow_pickle=False) as a:
        prof = np.asarray(a["points"], float)
    with np.load(args.reg_inputs, allow_pickle=False) as a:
        tgt_ifc = np.asarray(a["target"], float)
    g3 = np.array(json.loads(args.g3.read_text()), float)
    inv = np.linalg.inv(g3)
    lo_i, hi_i = tgt_ifc.min(0), tgt_ifc.max(0)
    corners = np.array([[x, y, z] for x in (lo_i[0], hi_i[0])
                        for y in (lo_i[1], hi_i[1]) for z in (lo_i[2], hi_i[2])])
    cr = apply_t(corners, inv)
    lo_r, hi_r = cr.min(0) - 0.5, cr.max(0) + 0.5
    prof_b = prof[np.all((prof >= lo_r) & (prof <= hi_r), axis=1)]
    print(f"professional building ref: {len(prof_b)} points", flush=True)

    # 2) căn thô: PCA + signed perms + scale search
    rec_core = dense_core(pts)
    if len(rec_core) > 100_000:
        rec_core = rec_core[rng.choice(len(rec_core), 100_000, replace=False)]
    ref_core = dense_core(prof_b)
    if len(ref_core) > 100_000:
        ref_core = ref_core[rng.choice(len(ref_core), 100_000, replace=False)]
    cr_, vr, wr = pca_frame(ref_core)
    cc, vc, wc = pca_frame(rec_core)
    s_base = float(np.sqrt(wr.sum() / wc.sum()))
    tree_ref = cKDTree(ref_core)
    best = None
    for f in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5):
        s0 = s_base * f
        for q in signed_permutations():
            R = vr @ q @ vc.T
            t = cr_ - s0 * (R @ cc)
            d, _ = tree_ref.query(s0 * (rec_core @ R.T) + t, workers=4)
            sc = float(np.mean(np.minimum(d, 2.0)))
            if best is None or sc < best[0]:
                best = (sc, s0, R, t)
    _, scale, R, t = best
    print(f"coarse score={best[0]:.3f} scale0={scale:.4f}", flush=True)

    # 3) ICP similarity thô
    cur = apply_sim(rec_core, scale, R, t)
    for it in range(40):
        d, idx = tree_ref.query(cur, workers=4)
        keep = d < np.quantile(d, 0.7 if it < 15 else 0.9)
        s2, r2, t2 = umeyama(cur[keep], ref_core[idx[keep]])
        if it < 15:
            s2, t2 = 1.0, ref_core[idx[keep]].mean(0) - r2 @ cur[keep].mean(0)
        else:
            s2 = float(np.clip(s2, 0.9, 1.1))
        cur = apply_sim(cur, s2, r2, t2)
        scale, R, t = s2 * scale, r2 @ R, s2 * (r2 @ t) + t2
    aligned = apply_sim(pts, scale, R, t)

    # 4) crop bbox khối nhà + outlier removal (mét)
    m = np.all((aligned >= lo_r) & (aligned <= hi_r), axis=1)
    aligned, col = aligned[m], col[m]
    tree_a = cKDTree(aligned)
    d, _ = tree_a.query(aligned, k=9, workers=4)
    md = d[:, 1:].mean(1)
    keep = md < 3 * np.median(md)
    aligned, col = aligned[keep], col[keep]
    print(f"crop+outlier: -> {len(aligned)} points", flush=True)

    # 5) ICP similarity mịn vào khối nhà chuyên nghiệp
    tree_pb = cKDTree(prof_b)
    cur = aligned
    for it in range(40):
        d, idx = tree_pb.query(cur, workers=4)
        keep = d < np.quantile(d, 0.8)
        s2, r2, t2 = umeyama(cur[keep], prof_b[idx[keep]])
        s2 = float(np.clip(s2, 0.95, 1.05))
        cur = apply_sim(cur, s2, r2, t2)
        scale, R, t = s2 * scale, r2 @ R, s2 * (r2 @ t) + t2
    final_ref = cur
    normals = pca_normals(final_ref, k=16)

    # metrics overlap hai chiều với khối nhà chuyên nghiệp
    d_r2p, _ = tree_pb.query(final_ref, workers=4)
    tree_f = cKDTree(final_ref)
    d_p2r, _ = tree_f.query(prof_b, workers=4)
    metrics = {
        "n_input": n0, "n_after_veg": n1, "n_final": int(len(final_ref)),
        "scale": float(scale),
        "recon_to_prof_m": {"mean": float(d_r2p.mean()), "median": float(np.median(d_r2p)),
                            "p90": float(np.quantile(d_r2p, 0.9))},
        "prof_to_recon_m": {"mean": float(d_p2r.mean()), "median": float(np.median(d_p2r)),
                            "p90": float(np.quantile(d_p2r, 0.9))},
        "coverage_recon_at_0.2": float(np.mean(d_r2p <= 0.2)),
        "coverage_prof_at_0.2": float(np.mean(d_p2r <= 0.2)),
    }
    print(json.dumps(metrics, indent=1), flush=True)

    # xuất khung ref (eval đóng băng) + khung IFC (viewer)
    mesh = trimesh.Trimesh(vertices=final_ref, vertex_normals=normals,
                           vertex_colors=col, process=False)
    mesh.export(args.out_dir / "recon_aligned.ply")
    np.savez_compressed(args.out_dir / "recon_attributes.npz", points=final_ref,
                        normals=normals, colors=col.astype(float) / 255.0)
    ifc_pts = apply_t(final_ref, g3)
    trimesh.Trimesh(vertices=ifc_pts, vertex_normals=normals @ g3[:3, :3].T,
                    vertex_colors=col, process=False).export(args.out_dir / "recon_ifc.ply")
    T = np.eye(4); T[:3, :3] = scale * R; T[:3, 3] = t
    (args.out_dir / "transform_recon_to_ref.json").write_text(json.dumps(T.tolist(), indent=1))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()