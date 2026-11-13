"""Căn chỉnh cloud dựng lại (COLMAP) vào hệ cloud gốc week26.ply và đo chất lượng.

Similarity transform (scale + R + t): PCA thô (24 phép quay trục) + ICP có scale.
Xuất: recon_aligned.ply, recon_attributes.npz (points/normals/colors cùng thứ tự),
transform_recon_to_ref.json, metrics.json (Chamfer + coverage hai chiều).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def clean_cloud(pts, normals, colors, radius=80.0, k=8):
    med = np.median(pts, axis=0)
    keep = np.linalg.norm(pts - med, axis=1) < radius
    p = pts[keep]
    tree = cKDTree(p)
    d, _ = tree.query(p, k=k + 1, workers=4)
    md = d[:, 1:].mean(axis=1)
    keep2 = md < 3 * np.median(md)
    idx = np.flatnonzero(keep)[keep2]
    return pts[idx], normals[idx] if normals is not None else None, colors[idx]


def pca_frame(pts):
    med = np.median(pts, axis=0)
    rad = np.linalg.norm(pts - med, axis=1)
    core = pts[rad <= np.quantile(rad, 0.98)]
    c = core.mean(axis=0)
    cov = np.cov((core - c).T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    v = v[:, order]
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1
    return c, v, w[order]


def signed_permutations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            q = np.zeros((3, 3))
            for i, p in enumerate(perm):
                q[i, p] = signs[i]
            if np.linalg.det(q) > 0:
                yield q


def umeyama(src, dst):
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - cs, dst - cd
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    scale = float((d * np.diag(s)).sum() / ((xs ** 2).sum() / len(src)))
    t = cd - scale * r @ cs
    return scale, r, t


def apply_sim(pts, scale, r, t):
    return scale * (pts @ r.T) + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, required=True)
    ap.add_argument("--ref-attributes", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--icp-iters", type=int, default=60)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    from trimesh.exchange.ply import load_ply
    with open(args.recon, "rb") as h:
        data = load_ply(h)
    rec = np.asarray(data["vertices"], dtype=float)
    raw_n = data.get("vertex_normals", data.get("normals"))
    rec_n = np.asarray(raw_n, dtype=float) if raw_n is not None else None
    if rec_n is not None and len(rec_n) == len(rec):
        rec_n = rec_n / np.maximum(np.linalg.norm(rec_n, axis=1)[:, None], 1e-12)
    else:
        rec_n = None
    raw_c = data.get("vertex_colors")
    if raw_c is not None:
        rec_col = np.asarray(raw_c, dtype=np.uint8)[:, :3]
    elif data.get("red") is not None:
        rec_col = np.stack([data[k] for k in ("red", "green", "blue")], axis=1).astype(np.uint8)
    else:
        rec_col = np.full((len(rec), 3), 128, dtype=np.uint8)
    if len(rec) > 500_000:
        sel = rng.choice(len(rec), 500_000, replace=False)
        rec = rec[sel]
        rec_n = rec_n[sel] if rec_n is not None else None
        rec_col = rec_col[sel]
    with np.load(args.ref_attributes, allow_pickle=False) as a:
        ref = np.asarray(a["points"], dtype=float)

    n_raw = len(rec)
    rec, rec_n, rec_col = clean_cloud(rec, rec_n, rec_col)
    print(f"clean: {n_raw} -> {len(rec)} points", flush=True)

    if rec_n is not None:
        n = rec_n.copy()
        flip = n[np.arange(len(n)), np.argmax(np.abs(n), axis=1)] < 0
        n[flip] *= -1
        w, v = np.linalg.eigh(n.T @ n)
        d1 = v[:, -1]
        resid = n - (n @ d1)[:, None] * d1
        w2, v2 = np.linalg.eigh(resid.T @ resid)
        d2 = v2[:, -1]
        d2 -= d1 * (d2 @ d1)
        d2 /= np.linalg.norm(d2)
        d3 = np.cross(d1, d2)
        rn = np.stack([d1, d2, d3])
        rec = rec @ rn.T
        rec_n = rec_n @ rn.T
        print("normal-frame applied", flush=True)

    def dense_core(pts, q=0.5, k=8):
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=k + 1, workers=4)
        md = d[:, 1:].mean(axis=1)
        return pts[md <= np.quantile(md, q)]

    rec_core = dense_core(rec)
    if len(rec_core) > 100_000:
        rec_core = rec_core[rng.choice(len(rec_core), 100_000, replace=False)]
    ref_sub = ref[rng.choice(len(ref), min(200_000, len(ref)), replace=False)]
    ref_core = dense_core(ref_sub)
    print(f"cores: rec={len(rec_core)} ref={len(ref_core)}", flush=True)

    cr, vr, wr = pca_frame(ref_core)
    cc, vc, wc = pca_frame(rec_core)
    s0 = float(np.sqrt(wr.sum() / wc.sum()))
    tree_ref = cKDTree(ref_core)
    best = None
    for q in signed_permutations():
        r = vr @ q @ vc.T
        t = cr - s0 * (r @ cc)
        d, _ = tree_ref.query(s0 * (rec_core @ r.T) + t, workers=4)
        score = float(np.mean(np.minimum(d, 2.0)))
        if best is None or score < best[0]:
            best = (score, r, t)
    _, r0, t0 = best
    print(f"coarse score={best[0]:.3f} m scale0={s0:.4f}", flush=True)

    cur = s0 * (rec_core @ r0.T) + t0
    scale, r, t = s0, r0, t0
    for it in range(args.icp_iters):
        d, idx = tree_ref.query(cur, workers=4)
        if it < 15:
            keep = d < np.quantile(d, 0.7)
            _, r2, _ = umeyama(cur[keep], ref_core[idx[keep]])
            s2 = 1.0
            t2 = ref_core[idx[keep]].mean(axis=0) - r2 @ cur[keep].mean(axis=0)
        else:
            keep = d < np.quantile(d, 0.9)
            s2, r2, t2 = umeyama(cur[keep], ref_core[idx[keep]])
            s2 = float(np.clip(s2, 0.9, 1.1))
        cur = apply_sim(cur, s2, r2, t2)
        scale, r, t = s2 * scale, r2 @ r, s2 * (r2 @ t) + t2
        if it >= 15 and abs(s2 - 1) < 1e-5 and np.linalg.norm(t2) < 1e-4:
            break
    print(f"icp iters={it + 1} scale={scale:.4f}", flush=True)

    aligned = apply_sim(rec, scale, r, t)
    if rec_n is not None:
        rec_n = rec_n @ r.T
        rec_n /= np.linalg.norm(rec_n, axis=1)[:, None]

    tree_rec = cKDTree(aligned)
    tree_ref_full = cKDTree(ref)
    d_r2f, _ = tree_ref_full.query(aligned, workers=4)
    d_f2r, _ = tree_rec.query(ref, workers=4)
    taus = [0.05, 0.1, 0.2, 0.3]
    metrics = {
        "n_recon": int(len(aligned)),
        "n_ref": int(len(ref)),
        "scale": scale,
        "recon_to_ref_m": {
            "mean": float(d_r2f.mean()), "median": float(np.median(d_r2f)),
            "p90": float(np.quantile(d_r2f, 0.9)),
        },
        "ref_to_recon_m": {
            "mean": float(d_f2r.mean()), "median": float(np.median(d_f2r)),
            "p90": float(np.quantile(d_f2r, 0.9)),
        },
        "chamfer_mean_m": float(0.5 * (d_r2f.mean() + d_f2r.mean())),
        "coverage_ref_at_tau": {str(t): float(np.mean(d_f2r <= t)) for t in taus},
        "coverage_recon_at_tau": {str(t): float(np.mean(d_r2f <= t)) for t in taus},
        "bbox_aligned_min": aligned.min(axis=0).tolist(),
        "bbox_aligned_max": aligned.max(axis=0).tolist(),
    }

    transform = np.eye(4)
    transform[:3, :3] = scale * r
    transform[:3, 3] = t
    (args.out_dir / "transform_recon_to_ref.json").write_text(json.dumps(transform.tolist(), indent=1))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))

    out_mesh = trimesh.Trimesh(vertices=aligned, vertex_normals=rec_n,
                               vertex_colors=rec_col, process=False)
    out_mesh.export(args.out_dir / "recon_aligned.ply")
    np.savez_compressed(
        args.out_dir / "recon_attributes.npz",
        points=aligned,
        normals=rec_n if rec_n is not None else np.zeros_like(aligned),
        colors=rec_col.astype(np.float64) / 255.0,
    )
    print(json.dumps(metrics, indent=1), flush=True)


if __name__ == "__main__":
    main()