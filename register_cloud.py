import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from progress_baseline import sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def voxel_points(points, size):
    _, indices = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]


def rigid_fit(source, target):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def transformed(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def building_frames(points, normals, bounds):
    low, high = np.asarray(bounds, dtype=float).reshape(2, 3)
    roi = np.all((points >= low) & (points <= high), axis=1)
    axes = []
    for axis in range(3):
        mask = roi & (np.abs(normals[:, axis]) > 0.96)
        if mask.sum() < 50:
            raise ValueError("Insufficient planar normals for frame estimation")
        _, _, vt = np.linalg.svd(normals[mask], full_matrices=False)
        direction = vt[0]
        if direction[axis] < 0:
            direction = -direction
        axes.append(direction)
    vx, vy, vz = axes
    vx -= vz * (vx @ vz)
    vx /= np.linalg.norm(vx)
    vy -= vz * (vy @ vz)
    vy -= vx * (vy @ vx)
    vy /= np.linalg.norm(vy)
    frames = []
    for horizontal in (vx, -vx, vy, -vy):
        down = -vz
        frames.append(np.stack([horizontal, np.cross(down, horizontal), down]))
    return frames


def trimmed_icp(source, target, tree, initial, trim, iterations):
    matrix = initial.copy()
    previous = float("inf")
    for iteration in range(iterations):
        current = transformed(source, matrix)
        distances, indices = tree.query(current, workers=2)
        selected = distances <= np.quantile(distances, trim)
        update = rigid_fit(current[selected], target[indices[selected]])
        matrix = update @ matrix
        error = float(np.sqrt(np.mean(distances[selected] ** 2)))
        if abs(previous - error) < 1e-6:
            break
        previous = error
    distances, _ = tree.query(transformed(source, matrix), workers=2)
    selected = distances <= np.quantile(distances, trim)
    return matrix, {
        "trimmed_rmse_m": float(np.sqrt(np.mean(distances[selected] ** 2))),
        "median_distance_m": float(np.median(distances)),
        "p90_distance_m": float(np.quantile(distances, 0.9)),
        "fraction_within_0_2m": float(np.mean(distances <= 0.2)),
        "iterations": iteration + 1,
    }


def register(args):
    started = time.monotonic()
    if not np.isfinite([args.voxel, args.trim, *args.scales, *args.z_offsets]).all() or args.voxel <= 0 or not 0 < args.trim <= 1 or args.yaw_steps < 1 or args.iterations < 1 or min(args.scales) <= 0:
        raise ValueError("Invalid registration parameters")
    with np.load(args.samples, allow_pickle=False) as archive:
        cloud = archive["cloud"]
        bim = archive["bim"]
    selected = np.ones(len(cloud), dtype=bool)
    if args.z_min is not None:
        selected &= cloud[:, 2] >= args.z_min
    if args.z_max is not None:
        selected &= cloud[:, 2] <= args.z_max
    if args.source_bounds:
        bounds = np.asarray(args.source_bounds).reshape(2, 3)
        if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            raise ValueError("Source bounds must be finite minima followed by larger maxima")
        selected &= np.all((cloud >= bounds[0]) & (cloud <= bounds[1]), axis=1)
    source = voxel_points(cloud[selected], args.voxel)
    target = voxel_points(bim, args.voxel)
    if len(source) < 20 or len(target) < 20:
        raise ValueError("Insufficient points for registration")
    cluster_counts = {}
    if args.cluster_radius:
        labels = DBSCAN(eps=args.cluster_radius, min_samples=4, n_jobs=2).fit_predict(source)
        cluster_counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels) if label >= 0}
        if not cluster_counts:
            raise ValueError("No connected source cluster found")
        source = source[labels == max(cluster_counts, key=cluster_counts.get)]
    tree = cKDTree(target)
    source_center = np.median(source, axis=0)
    target_center = np.median(target, axis=0)
    if args.normal_frame:
        if not args.source_bounds:
            raise ValueError("Normal-frame initialization requires source bounds")
        with np.load(args.normal_frame, allow_pickle=False) as archive:
            rotations = [(f"frame{index}", frame) for index, frame in enumerate(building_frames(archive["points"], archive["normals"], args.source_bounds))]
    else:
        gravity_flip = np.diag([1, -1, -1]) if args.flip_z else np.eye(3)
        rotations = []
        for yaw_index in range(args.yaw_steps):
            angle = yaw_index * 2 * np.pi / args.yaw_steps
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ gravity_flip
            rotations.append((f"yaw{np.degrees(angle):.0f}", rotation))
    candidates = []
    for scale in args.scales:
        for rotation_label, rotation in rotations:
            for z_offset in args.z_offsets:
                initial = np.eye(4)
                initial[:3, :3] = scale * rotation
                initial[:3, 3] = target_center - scale * rotation @ source_center
                initial[2, 3] += z_offset
                matrix, scores = trimmed_icp(source, target, tree, initial, args.trim, args.iterations)
                candidate = {
                    "initial_rotation": rotation_label,
                    "initial_z_offset_m": z_offset, "scale": scale,
                    "transform": matrix.tolist(), **scores,
                }
                candidates.append(candidate)
                print(json.dumps({key: value for key, value in candidate.items() if key != "transform"}), flush=True)
    candidates.sort(key=lambda candidate: candidate["trimmed_rmse_m"])
    coarse_best = np.asarray(candidates[0]["transform"])
    refined, refined_scores = trimmed_icp(source, bim, cKDTree(bim), coarse_best, args.trim, args.iterations)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidate_transform.json").write_text(json.dumps(refined.tolist(), indent=2) + "\n", encoding="utf-8")
    aligned = transformed(source, refined)
    np.savez_compressed(args.output / "registration_inputs.npz", source=source, target=target, aligned=aligned)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        axis.scatter(target[:, a], target[:, b], s=0.5, alpha=0.4, color="#cc433d", label="BIM")
        axis.scatter(aligned[:, a], aligned[:, b], s=0.6, alpha=0.5, color="#168da4", label="Aligned scan cluster")
        axis.set_xlabel(f"{'XYZ'[a]} (m)")
        axis.set_ylabel(f"{'XYZ'[b]} (m)")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    axes[0].legend(markerscale=6)
    fig.suptitle("Unverified registration candidate: requires independent control checks")
    fig.tight_layout()
    fig.savefig(args.output / "candidate_alignment.png", dpi=160)
    plt.close(fig)
    report = {
        "method": "fixed-scale-multistart-trimmed-point-to-point-icp-v2",
        "samples_path": str(args.samples), "samples_sha256": sha256(args.samples),
        "voxel_raw_units_source_m_target": args.voxel, "trim_fraction": args.trim,
        "source_z_filter_raw_units": [args.z_min, args.z_max], "cluster_radius_raw_units": args.cluster_radius,
        "source_bounds_raw_units": args.source_bounds, "flip_z": args.flip_z,
        "normal_frame_path": str(args.normal_frame) if args.normal_frame else None,
        "normal_frame_sha256": sha256(args.normal_frame) if args.normal_frame else None,
        "candidate_scales": args.scales, "chosen_scale": candidates[0]["scale"],
        "cluster_counts": cluster_counts, "n_source": len(source), "n_target": len(target),
        "yaw_steps": args.yaw_steps, "z_offsets_m": args.z_offsets,
        "all_candidates": candidates, "refined_scores": refined_scores,
        "refined_transform": refined.tolist(), "elapsed_seconds": time.monotonic() - started,
        "independently_validated": False,
        "limitations": [
            "Scale and gravity orientation are hypotheses searched explicitly; low residual does not verify either.",
            "Repeated floors and partial construction may yield a geometrically plausible but incorrect local optimum.",
            "Residuals against sampled BIM are optimization diagnostics, not registration ground truth or progress accuracy.",
        ],
    }
    (args.output / "registration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"refined_scores": refined_scores, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.25)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--flip-z", action="store_true")
    parser.add_argument("--source-bounds", type=float, nargs=6)
    parser.add_argument("--normal-frame", type=Path)
    parser.add_argument("--trim", type=float, default=0.6)
    parser.add_argument("--yaw-steps", type=int, default=18)
    parser.add_argument("--z-offsets", type=float, nargs="+", default=[-3.0, 0.0, 3.0])
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--cluster-radius", type=float, default=0.75)
    parser.add_argument("--iterations", type=int, default=60)
    register(parser.parse_args())


if __name__ == "__main__":
    main()
