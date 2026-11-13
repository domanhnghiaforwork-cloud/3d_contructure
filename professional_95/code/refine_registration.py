"""Tinh chỉnh đăng ký cloud chuyên nghiệp week26.ply -> IFC (khung IFC).

Khởi tạo từ G3 đóng băng (candidate_transform.json), rồi ICP point-to-plane
(trimmed, rigid: chỉ R+t, khóa scale vì cloud chuyên nghiệp đã đúng thước) against
mẫu bề mặt IFC (registration_inputs.npz['target']). Giảm lệch hướng nhỏ còn dư.

KHÔNG đụng G3 đóng băng: xuất transform tinh chỉnh RIÊNG
(candidate_transform_refined.json) + metrics trước/sau. Dùng cho hiển thị viewer
và làm mỏ neo căn cloud video; đánh giá đóng băng vẫn dùng G3 gốc.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]


def apply_t(pts, t4):
    return pts @ t4[:3, :3].T + t4[:3, 3]


def pca_normals(pts, k=16):
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k, workers=4)
    nb = pts[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    return n


def expmap(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rot_angle_deg(r):
    c = (np.trace(r) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def dist_stats(src, tree):
    d, _ = tree.query(src, workers=4)
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "p90": float(np.quantile(d, 0.9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path,
                    default=ROOT / "artifacts/week26_registration_v3/registration_inputs.npz")
    ap.add_argument("--g3", type=Path,
                    default=ROOT / "artifacts/week26_registration_v3/candidate_transform.json")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "artifacts/week26_registration_refined")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--trim", type=float, default=0.8)
    ap.add_argument("--crop-margin", type=float, default=1.0,
                    help="crop source về bbox mẫu IFC (+margin m) để chỉ tinh chỉnh khối nhà")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.inputs, allow_pickle=False) as a:
        src_ref = np.asarray(a["source"], float)
        tgt = np.asarray(a["target"], float)
    g3 = np.array(json.loads(args.g3.read_text()), float)

    tree = cKDTree(tgt)
    n_tgt = pca_normals(tgt)
    cur = apply_t(src_ref, g3)
    full_before = dist_stats(cur, tree)
    lo, hi = tgt.min(axis=0) - args.crop_margin, tgt.max(axis=0) + args.crop_margin
    bmask = np.all((cur >= lo) & (cur <= hi), axis=1)
    cur = cur[bmask]
    before = dist_stats(cur, tree)
    print(f"full before G3: {full_before}", flush=True)
    print(f"building-only n={len(cur)} before G3: {before}", flush=True)

    delta = np.eye(4)
    for it in range(args.iters):
        d, idx = tree.query(cur, workers=4)
        keep = d < np.quantile(d, args.trim)
        p = cur[keep]
        q = tgt[idx[keep]]
        n = n_tgt[idx[keep]]
        # point-to-plane linearized: residual = (p-q).n + r.(p x n) + t.n
        pn = np.cross(p, n)
        A = np.concatenate([pn, n], axis=1)
        b = -np.einsum("ij,ij->i", p - q, n)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        r, t = x[:3], x[3:]
        R = expmap(r)
        step = np.eye(4)
        step[:3, :3] = R
        step[:3, 3] = t
        cur = apply_t(cur, step)
        delta = step @ delta
        if np.linalg.norm(r) < 1e-6 and np.linalg.norm(t) < 1e-5:
            break
    after = dist_stats(cur, tree)
    refined = delta @ g3
    ang = rot_angle_deg(delta[:3, :3])
    trans = float(np.linalg.norm(delta[:3, 3]))
    print(f"after refine: {after}", flush=True)
    print(f"delta rotation={ang:.3f} deg  translation={trans:.4f} m  iters={it+1}", flush=True)

    metrics = {"full_before_g3_m": full_before,
               "building_before_g3_m": before, "building_after_refined_m": after,
               "delta_rotation_deg": ang, "delta_translation_m": trans,
               "iters": it + 1, "n_source": int(len(src_ref)),
               "n_building": int(len(cur)), "n_target": int(len(tgt))}
    (args.out_dir / "candidate_transform_refined.json").write_text(json.dumps(refined.tolist(), indent=1))
    (args.out_dir / "refine_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1), flush=True)


if __name__ == "__main__":
    main()