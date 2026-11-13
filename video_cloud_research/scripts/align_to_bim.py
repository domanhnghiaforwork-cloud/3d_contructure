"""Căn chỉnh cloud dựng lại vào hệ IFC/BIM với cổng chất lượng (inlier >= 0.7).

Đa khởi tạo: lưới scale x 24 hoán vị trục, similarity ICP trim, chấm inlier
0.15 m trên lõi mật độ cao; refine rồi xuất aligned attributes + transform.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import align_and_compare as ac

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gate", type=float, default=0.7)
    ap.add_argument("--scale-base", type=float, default=None)
    ap.add_argument("--scale-lo", type=float, default=0.5)
    ap.add_argument("--scale-hi", type=float, default=2.0)
    ap.add_argument("--bim", type=Path,
                    default=ROOT / "artifacts/week26_registration_v3/registration_inputs.npz")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(42)

    from trimesh.exchange.ply import load_ply
    with open(args.recon, "rb") as h:
        data = load_ply(h)
    rec = np.asarray(data["vertices"], dtype=float)
    raw_n = data.get("vertex_normals", data.get("normals"))
    n = np.asarray(raw_n, dtype=float) if raw_n is not None else None
    if n is not None:
        n = n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    raw_c = data.get("vertex_colors")
    c = np.asarray(raw_c, np.uint8)[:, :3] if raw_c is not None else np.full((len(rec), 3), 128, np.uint8)
    if len(rec) > 500_000:
        sel = rng.choice(len(rec), 500_000, replace=False)
        rec, c = rec[sel], c[sel]
        n = n[sel] if n is not None else None
    rec, n, c = ac.clean_cloud(rec, n, c)

    if n is not None:
        nn = n.copy()
        flip = nn[np.arange(len(nn)), np.argmax(np.abs(nn), axis=1)] < 0
        nn[flip] *= -1
        w, v = np.linalg.eigh(nn.T @ nn)
        d1 = v[:, -1]
        res = nn - (nn @ d1)[:, None] * d1
        w2, v2 = np.linalg.eigh(res.T @ res)
        d2 = v2[:, -1]
        d2 -= d1 * (d2 @ d1)
        d2 /= np.linalg.norm(d2)
        rn = np.stack([d1, d2, np.cross(d1, d2)])
        rec = rec @ rn.T
        n = n @ rn.T

    def dense_core(pts, q=0.5, k=8):
        tr = cKDTree(pts)
        d, _ = tr.query(pts, k=k + 1, workers=4)
        md = d[:, 1:].mean(1)
        return pts[md <= np.quantile(md, q)]

    core = dense_core(rec)
    if len(core) > 60_000:
        core = core[rng.choice(len(core), 60_000, replace=False)]
    bim = np.load(args.bim)["target"]
    tree = cKDTree(bim)
    tc = np.median(bim, axis=0)
    sc = np.median(core, axis=0)
    rad_c = np.quantile(np.linalg.norm(core - sc, axis=1), 0.9)
    rad_b = np.quantile(np.linalg.norm(bim - tc, axis=1), 0.9)
    s_base = float(rad_b / rad_c)
    if args.scale_base is not None:
        s_base = args.scale_base
    cr, vr, _ = ac.pca_frame(bim)
    cc, vc, _ = ac.pca_frame(core)
    lo, hi = args.scale_lo * s_base, args.scale_hi * s_base
    print(f"scale base {s_base:.3f} (bounds {lo:.3f}..{hi:.3f})", flush=True)

    best = None
    for f in (0.7, 0.85, 1.0, 1.2, 1.4):
        s0 = s_base * f
        for q in ac.signed_permutations():
            r = vr @ q @ vc.T
            t = tc - s0 * (r @ sc)
            cur = s0 * (core @ r.T) + t
            scale, R, T = s0, r, t
            for it in range(15):
                d, idx = tree.query(cur, workers=4)
                keep = d < np.quantile(d, 0.6)
                s2, r2, t2 = ac.umeyama(cur[keep], bim[idx[keep]])
                s2 = float(np.clip(s2, 0.85, 1.15))
                ns = float(np.clip(s2 * scale, lo, hi))
                s2 = ns / scale
                cur = ac.apply_sim(cur, s2, r2, t2)
                scale, R, T = ns, r2 @ R, s2 * (r2 @ T) + t2
            d, _ = tree.query(cur, workers=4)
            score = float(np.mean(d <= 0.15))
            if best is None or score > best[0]:
                best = (score, scale, R, T)
        print(f"scale {s0}: best inlier {best[0]:.3f}", flush=True)

    score, scale, R, T = best
    best_ref = (score, scale, R, T)
    for it in range(40):
        cur = ac.apply_sim(core, scale, R, T)
        d, idx = tree.query(cur, workers=4)
        sc_it = float(np.mean(d <= 0.15))
        if sc_it > best_ref[0]:
            best_ref = (sc_it, scale, R, T)
        keep = d < np.quantile(d, 0.7)
        s2, r2, t2 = ac.umeyama(cur[keep], bim[idx[keep]])
        s2 = float(np.clip(s2, 0.9, 1.1))
        ns = float(np.clip(s2 * scale, lo, hi))
        s2 = ns / scale
        scale, R, T = ns, r2 @ R, s2 * (r2 @ T) + t2
        if abs(s2 - 1) < 1e-5 and np.linalg.norm(t2) < 1e-4:
            break
    score, scale, R, T = best_ref

    al = ac.apply_sim(rec, scale, R, T)
    core_al = ac.apply_sim(core, scale, R, T)
    nn2 = n @ R.T if n is not None else np.zeros_like(al)
    d_core, _ = tree.query(core_al, workers=4)
    inlier_core = float(np.mean(d_core <= 0.15))
    d, _ = tree.query(al, workers=4)
    frac = {str(x): float(np.mean(d <= x)) for x in (0.1, 0.2, 0.3)}
    db, _ = cKDTree(al).query(bim, workers=4)
    coverage_bim = {str(x): float(np.mean(db <= x)) for x in (0.1, 0.2, 0.3)}
    scale_ok = bool(lo <= scale <= hi)
    gate_passed = bool(scale_ok and inlier_core >= args.gate
                       and frac["0.2"] >= 0.5 and coverage_bim["0.2"] >= 0.25)
    metrics = {
        "inlier_core": inlier_core,
        "scale": scale,
        "scale_ok": scale_ok,
        "gate_passed": gate_passed,
        "frac_recon_within": frac,
        "coverage_bim": coverage_bim,
        "elapsed_s": round(time.time() - t0, 1),
    }
    M = np.eye(4)
    M[:3, :3] = scale * R
    M[:3, 3] = T
    (args.out_dir / "transform_recon_to_ifc.json").write_text(json.dumps(M.tolist(), indent=1))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez_compressed(args.out_dir / "recon_attributes.npz",
                        points=al, normals=nn2, colors=c.astype(float) / 255)
    import trimesh
    trimesh.Trimesh(vertices=al, vertex_normals=nn2, vertex_colors=c,
                    process=False).export(args.out_dir / "recon_aligned.ply")
    print(json.dumps(metrics, indent=1), flush=True)
    if not metrics["gate_passed"]:
        raise SystemExit(
            f"QUALITY GATE FAILED: inlier {inlier_core:.3f} (>= {args.gate}), "
            f"scale_ok {scale_ok}, frac@0.2 {frac['0.2']:.3f} (>= 0.5), "
            f"coverage_bim@0.2 {coverage_bim['0.2']:.3f} (>= 0.25)")


if __name__ == "__main__":
    main()