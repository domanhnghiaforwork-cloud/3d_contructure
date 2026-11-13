import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.unit
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

from progress_baseline import DEFAULT_TYPES, load_cloud, sample_surface, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def inspect(args):
    started = time.monotonic()
    if not np.isfinite(args.cloud_scale) or args.cloud_scale <= 0 or args.max_points < 1 or args.samples_per_element < 1:
        raise ValueError("Scale and sample counts must be finite and positive")
    rng = np.random.default_rng(args.seed)
    cloud = load_cloud(args.cloud) * args.cloud_scale
    cloud_count = len(cloud)
    cloud_bounds = [cloud.min(axis=0).tolist(), cloud.max(axis=0).tolist()]
    if len(cloud) > args.max_points:
        cloud = cloud[rng.choice(len(cloud), args.max_points, replace=False)]
    transform = np.eye(4)
    if args.transform:
        transform = np.asarray(json.loads(args.transform.read_text()), dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Expected a finite affine 4x4 cloud-to-IFC transform")
        cloud = cloud @ transform[:3, :3].T + transform[:3, 3]
    model = ifcopenshell.open(str(args.ifc))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    elements = {element.GlobalId: element for entity_type in args.types for element in model.by_type(entity_type)}
    rows = []
    failures = []
    samples = []
    for index, (guid, element) in enumerate(sorted(elements.items()), 1):
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
            vertices = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
            faces = np.asarray(shape.geometry.faces, dtype=int).reshape(-1, 3)
            points, area = sample_surface(vertices, faces, args.samples_per_element, rng)
            samples.append(points)
            bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            rows.append({
                "ifc_guid": guid, "ifc_type": element.is_a(), "name": element.Name or "",
                "area_m2": area,
                **{f"{limit}_{axis}": float(bounds[side, dimension]) for side, limit in enumerate(("min", "max")) for dimension, axis in enumerate("xyz")},
            })
        except (RuntimeError, ValueError) as error:
            failures.append({"ifc_guid": guid, "error": str(error)})
        if index % 200 == 0:
            print(f"Inspected {index}/{len(elements)} elements", flush=True)
    if not samples:
        raise ValueError("No usable IFC surface geometry")
    bim = np.concatenate(samples)
    tree = cKDTree(cloud)
    residual, _ = tree.query(bim, workers=2)
    bim_min, bim_max = bim.min(axis=0), bim.max(axis=0)
    margin = np.maximum((bim_max - bim_min) * 0.15, 2.0)
    cloud_near_model = cloud[np.all((cloud >= bim_min - margin) & (cloud <= bim_max + margin), axis=1)]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "ifc_elements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / "inspection_samples.npz", cloud=cloud, bim=bim)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, dimensions in zip(axes, ((0, 1), (0, 2), (1, 2))):
        a, b = dimensions
        axis.scatter(cloud[:, a], cloud[:, b], s=0.15, alpha=0.5, color="#168da4", label="Observed cloud")
        axis.scatter(bim[:, a], bim[:, b], s=0.3, alpha=0.5, color="#cf413c", label="Planned BIM")
        axis.set_xlabel(f"{'XYZ'[a]} (m)")
        axis.set_ylabel(f"{'XYZ'[b]} (m)")
        axis.set_xlim(bim_min[a] - margin[a], bim_max[a] + margin[a])
        axis.set_ylim(bim_min[b] - margin[b], bim_max[b] + margin[b])
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    axes[0].legend(markerscale=8)
    fig.suptitle(f"Alignment diagnostic: {args.cloud.name}\nNot a construction-status evaluation")
    fig.tight_layout()
    fig.savefig(args.output / "alignment_overview.png", dpi=160)
    plt.close(fig)
    report = {
        "purpose": "Geometry and coordinate diagnostics, not ground-truth construction labels",
        "ifc_path": str(args.ifc), "ifc_sha256": sha256(args.ifc),
        "cloud_path": str(args.cloud), "cloud_sha256": sha256(args.cloud),
        "ifc_schema": model.schema,
        "ifc_length_unit_to_m": ifcopenshell.util.unit.calculate_unit_scale(model),
        "cloud_scale": args.cloud_scale,
        "cloud_to_ifc_transform": transform.tolist(),
        "input_cloud_points": cloud_count,
        "inspection_cloud_points": len(cloud),
        "input_cloud_bounds_scaled_m": cloud_bounds,
        "inspection_cloud_points_near_bim": len(cloud_near_model),
        "bim_sample_bounds_m": [bim_min.tolist(), bim_max.tolist()],
        "n_selected_elements": len(elements), "n_geometry_ok": len(rows),
        "ifc_types": dict(Counter(element.is_a() for element in elements.values())),
        "storeys": [{"name": storey.Name, "elevation_ifc_units": storey.Elevation} for storey in model.by_type("IfcBuildingStorey")],
        "bim_to_sampled_cloud_distance_quantiles_m": dict(zip(("p10", "p50", "p90", "p99"), np.quantile(residual, [0.1, 0.5, 0.9, 0.99]).tolist())),
        "geometry_failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "caveat": "Residuals use subsampled points and may include genuinely unbuilt or occluded geometry; do not choose progress labels from these diagnostics.",
    }
    (args.output / "inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "geometry_failures"}, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transform", type=Path)
    parser.add_argument("--cloud-scale", type=float, default=1.0)
    parser.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES))
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--samples-per-element", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    inspect(parser.parse_args())


if __name__ == "__main__":
    main()
