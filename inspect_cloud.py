import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import trimesh

from progress_baseline import sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def inspect(args):
    cloud = trimesh.load(args.cloud, process=False)
    points = np.asarray(cloud.vertices)
    data = cloud.metadata["_ply_raw"]["vertex"]["data"]
    colors = np.column_stack([data[f"diffuse_{color}"] for color in ("red", "green", "blue")]).reshape(-1, 3) / 255
    normals = np.column_stack([data[f"n{axis}"] for axis in "xyz"]).reshape(-1, 3)
    if len(points) != len(colors) or len(points) != len(normals):
        raise ValueError("PLY attributes do not match vertices")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(points), min(args.max_points, len(points)), replace=False)
    selected, rgb = points[indices], colors[indices]
    bounds = np.quantile(points, [0.01, 0.99], axis=0)
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, (a, b) in zip(axes[0], ((0, 1), (0, 2), (1, 2))):
        axis.scatter(selected[:, a], selected[:, b], c=rgb, s=0.4)
        axis.set_xlabel("XYZ"[a])
        axis.set_ylabel("XYZ"[b])
        axis.set_xlim(bounds[:, a])
        axis.set_ylim(bounds[:, b])
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    axes[1, 0].scatter(selected[:, 0], selected[:, 1], c=selected[:, 2], vmin=bounds[0, 2], vmax=bounds[1, 2], cmap="viridis", s=0.4)
    axes[1, 0].set_xlim(bounds[:, 0])
    axes[1, 0].set_ylim(bounds[:, 1])
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("XY colored by raw Z")
    axes[1, 1].hist(points[:, 2], bins=np.arange(bounds[0, 2], bounds[1, 2], 0.05))
    axes[1, 1].set_title("Raw Z distribution (0.05-unit bins)")
    for index, label in enumerate("XYZ"):
        axes[1, 2].hist(np.abs(normals[:, index]), bins=60, histtype="step", label=label)
    axes[1, 2].set_title("Absolute input normal components")
    axes[1, 2].legend()
    fig.suptitle("Raw scan diagnostics: coordinate scale and gravity NOT yet verified")
    fig.tight_layout()
    fig.savefig(args.output / "raw_cloud_overview.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    cuts = np.linspace(bounds[0, 2], bounds[1, 2], 7)
    for axis, low, high in zip(axes.ravel(), cuts[:-1], cuts[1:]):
        keep = (selected[:, 2] >= low) & (selected[:, 2] < high)
        axis.scatter(selected[keep, 0], selected[keep, 1], c=rgb[keep], s=1)
        axis.set_xlim(bounds[:, 0])
        axis.set_ylim(bounds[:, 1])
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.set_title(f"Raw Z [{low:.2f}, {high:.2f})")
    fig.tight_layout()
    fig.savefig(args.output / "raw_cloud_slices.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(args.output / "cloud_attributes.npz", points=points, colors=colors, normals=normals)
    report = {"source": str(args.cloud), "sha256": sha256(args.cloud), "n_points": len(points), "seed": args.seed, "n_plotted": len(indices), "quantile_levels": [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1], "coordinate_quantiles": np.quantile(points, [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1], axis=0).tolist(), "coordinate_units": "unverified"}
    (args.output / "raw_cloud.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42)
    inspect(parser.parse_args())
