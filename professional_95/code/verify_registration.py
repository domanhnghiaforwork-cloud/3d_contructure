import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks

from progress_baseline import sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plane_peaks(values, min_count):
    counts, edges = np.histogram(values, np.arange(values.min(), values.max() + 0.02, 0.02))
    peaks, _ = find_peaks(counts, distance=5, prominence=max(30, counts.max() * 0.05))
    return [(float((edges[k] + edges[k + 1]) / 2), int(counts[k])) for k in peaks if counts[k] >= min_count]


def verify(args):
    matrix = np.asarray(json.loads(args.transform.read_text(encoding="utf-8")), dtype=float)
    with np.load(args.attributes, allow_pickle=False) as archive:
        points, normals = archive["points"], archive["normals"]
    bounds = np.asarray(args.source_bounds, dtype=float).reshape(2, 3)
    roi = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
    p = points[roi] @ matrix[:3, :3].T + matrix[:3, 3]
    n = normals[roi] @ matrix[:3, :3].T
    n /= np.linalg.norm(n, axis=1)[:, None]
    faces = {0: [], 1: [], 2: []}
    with args.elements.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ifc_type"] in ("IfcWall", "IfcWallStandardCase"):
                for axis in (0, 1):
                    faces[axis] += [float(row[f"min_{'xyz'[axis]}"]), float(row[f"max_{'xyz'[axis]}"])]
            elif row["ifc_type"] == "IfcSlab":
                faces[2] += [float(row["min_z"]), float(row["max_z"])]
    faces = {axis: np.unique(np.round(values, 3)) for axis, values in faces.items()}
    report = {"transform_path": str(args.transform), "transform_sha256": sha256(args.transform), "attributes_sha256": sha256(args.attributes), "n_roi_points": int(roi.sum()), "axes": {}}
    deltas_all, weights_all = [], []
    for axis in range(3):
        aligned = np.abs(n[:, axis]) > 0.96
        values = p[aligned, axis]
        if len(values) < 100:
            continue
        nearest = np.min(np.abs(values[:, None] - faces[axis][None, :]), axis=1)
        peaks = []
        for offset, count in plane_peaks(values, args.min_peak_points):
            delta = float(np.min(np.abs(offset - faces[axis])))
            peaks.append({"offset_m": offset, "points": count, "nearest_bim_face_delta_m": delta})
        report["axes"]["xyz"[axis]] = {
            "n_aligned_points": int(aligned.sum()),
            "fraction_points_within_0_1m_of_bim_face": float(np.mean(nearest <= 0.1)),
            "fraction_points_within_0_2m_of_bim_face": float(np.mean(nearest <= 0.2)),
            "median_point_delta_m": float(np.median(nearest)),
            "peaks": peaks,
        }
        deltas_all.append(nearest)
        weights_all.append(int(aligned.sum()))
    z_peaks = [peak["offset_m"] for peak in report["axes"].get("z", {}).get("peaks", [])]
    report["z_peak_spacings_m"] = np.diff(sorted(z_peaks)).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    wall_xy = np.abs(n[:, 2]) < 0.2
    delta_xy = np.min(np.abs(p[wall_xy, 0][:, None] - faces[0][None, :]), axis=1)
    delta_xy = np.minimum(delta_xy, np.min(np.abs(p[wall_xy, 1][:, None] - faces[1][None, :]), axis=1))
    scatter = axes[0].scatter(p[wall_xy, 0], p[wall_xy, 1], c=np.clip(delta_xy, 0, 1), cmap="viridis_r", s=0.5)
    fig.colorbar(scatter, ax=axes[0], label="Distance to nearest BIM wall face (m)")
    with args.elements.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ifc_type"] in ("IfcWall", "IfcWallStandardCase"):
                x0, x1, y0, y1 = (float(row[k]) for k in ("min_x", "max_x", "min_y", "max_y"))
                axes[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#cc433d", lw=0.4, alpha=0.6)
    axes[0].set_aspect("equal")
    axes[0].set_title("Transformed scan wall points vs BIM wall footprints")
    axes[1].scatter(p[:, 0], p[:, 2], c=np.clip(np.min(np.abs(p[:, 2][:, None] - faces[2][None, :]), axis=1), 0, 1), cmap="viridis_r", s=0.5)
    for z in faces[2]:
        axes[1].axhline(z, color="#cc433d", lw=0.3, alpha=0.5)
    axes[1].set_aspect("equal")
    axes[1].set_title("XZ: scan vs BIM slab faces")
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "registration_control_check.png", dpi=170)
    plt.close(fig)
    (args.output / "registration_control_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "axes"}, indent=2))
    for axis, data in report["axes"].items():
        print(axis, {key: value for key, value in data.items() if key != "peaks"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--elements", type=Path, required=True)
    parser.add_argument("--source-bounds", type=float, nargs=6, required=True)
    parser.add_argument("--min-peak-points", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args())
