"""Căn chỉnh cứng (scale cố định) cloud metric vào BIM, tối ưu ĐỘ PHỦ BIM.

Dành cho cloud MASt3R metric (scale thật ~1): hình học cục bộ tốt nhưng chỉ phủ
một phần tòa nhà. Phép căn chỉnh partial-to-full: quét 24 hoán vị trục PCA x lưới
tịnh tiến dọc 2 trục ngang BIM, chấm coverage_bim@0.2, rồi ICP cứng keep-best.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import align_and_compare as ac

ROOT = Path(__file__).resolve().parents[2]


def coverage(tree_pts, query_pts, tau):
    d, _ = cKDTree(tree_pts).query(query_pts, workers=4)
    return float(np.mean(d <= tau))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.2)
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
    raw_c = data.get("vertex_colors")
    c = np.asarray(raw_c, np.uint8)[:, :3] if raw_c is not None else np.full((len(rec), 3), 128, np.uint8)
    rec, _, c = ac.clean_cloud(rec, None, c)
    if len(rec) > 150_000:
        sel = rng.choice(len(rec), 150_000, replace=False)
        rec, c = rec[sel], c[sel]

    bim = np.load(args.bim)["target"]
    tc = np.median(bim, axis=0)
    cr, vr, _ = ac.pca_frame(bim)
    cc, vc, _ = ac.pca_frame(rec)
    ext = np.quantile(bim, 0.95, axis=0) - np.quantile(bim, 0.05, axis=0)

    rec_s = rec[rng.choice(len(rec), min(30_000, len(rec)), replace=False)]
    bim_s = bim[rng.choice(len(bim), min(20_000, len(bim)), replace=False)]
    s = args.scale
    offsets = [np.array([a * ext[0], b * ext[1], 0.0]) for a in (-0.35, 0, 0.35) for b in (-0.35, 0, 0.35)]

    best = None
    for q in ac.signed_permutations():
        R = vr @ q @ vc.T
        rc = s * (rec_s @ R.T)
        base_t = tc - np.median(rc, axis=0)
        for off in offsets:
            t = base_t + vr @ off
            cur = rc + t
            cov = coverage(cur, bim_s, args.tau)
            if best is None or cov > best[0]:
                best = (cov, R, t)
    cov0, R, T = best
    print(f"coarse best coverage@{args.tau}: {cov0:.3f}", flush=True)

    best_ref = (cov0, R, T)
    for it in range(30):
        cur = s * (rec @ R.T) + T
        cov = coverage(cur, bim_s, args.tau)
        if cov > best_ref[0]:
            best_ref = (cov, R, T)
        d, idx = cKDTree(bim).query(cur, workers=4)
        keep = d < np.quantile(d, 0.6)
        _, r2, t2 = ac.umeyama(cur[keep], bim[idx[keep]])
        R, T = r2 @ R, r2 @ T + t2
    cov, R, T = best_ref

    al = s * (rec @ R.T) + T
    d, _ = cKDTree(bim).query(al, workers=4)
    frac = {str(x): float(np.mean(d <= x)) for x in (0.1, 0.2, 0.3)}
    db, _ = cKDTree(al).query(bim, workers=4)
    cov_bim = {str(x): float(np.mean(db <= x)) for x in (0.1, 0.2, 0.3)}
    metrics = {"scale": s, "coverage_bim": cov_bim, "frac_recon_within": frac,
               "elapsed_s": round(time.time() - t0, 1)}
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = T
    (args.out_dir / "transform_recon_to_ifc.json").write_text(json.dumps(M.tolist(), indent=1))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez_compressed(args.out_dir / "recon_attributes.npz",
                        points=al, normals=np.zeros_like(al), colors=c.astype(float) / 255)
    print(json.dumps(metrics, indent=1), flush=True)


if __name__ == "__main__":
    main()