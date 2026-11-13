import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import numpy as np
import trimesh
from scipy.spatial import cKDTree


DEFAULT_TYPES = ("IfcWall", "IfcSlab", "IfcRoof", "IfcBeam", "IfcColumn")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cloud(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            points = archive["points"]
    elif suffix == ".ply":
        points = np.asarray(trimesh.load(path, process=False).vertices)
    elif suffix in {".xyz", ".txt"}:
        points = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
    else:
        raise ValueError(f"Unsupported point-cloud format: {suffix}")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Point cloud must have nonempty shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates")
    return points


def sample_surface_with_normals(vertices, faces, count, rng):
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(crosses, axis=1) / 2
    area = float(areas.sum())
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Element has no nondegenerate surface triangles")
    choice = rng.choice(len(triangles), size=count, p=areas / area)
    selected = triangles[choice]
    normals = crosses[choice] / np.linalg.norm(crosses[choice], axis=1)[:, None]
    u = np.sqrt(rng.random(count))[:, None]
    v = rng.random(count)[:, None]
    points = (1 - u) * selected[:, 0] + u * (1 - v) * selected[:, 1] + u * v * selected[:, 2]
    return points, normals, area


def sample_surface(vertices, faces, count, rng):
    points, _, area = sample_surface_with_normals(vertices, faces, count, rng)
    return points, area


def normal_aware_coverage(samples, sample_normals, cloud, cloud_normals, tree, along_tol, lateral_tol):
    supported = np.zeros(len(samples), dtype=bool)
    for index, neighbors in enumerate(tree.query_ball_point(samples, r=along_tol + lateral_tol)):
        if not neighbors:
            continue
        delta = cloud[neighbors] - samples[index]
        along = delta @ sample_normals[index]
        lateral = np.linalg.norm(delta - along[:, None] * sample_normals[index], axis=1)
        orientation = np.abs(cloud_normals[neighbors] @ sample_normals[index])
        supported[index] = bool(np.any((np.abs(along) <= along_tol) & (lateral <= lateral_tol) & (orientation >= 0.5)))
    return float(supported.mean())


def competitive_coverage(samples, sample_normals, cloud, cloud_normals, along_tol, lateral_tol, workers):
    sample_tree = cKDTree(samples)
    supported = np.zeros(len(samples), dtype=bool)
    nearest = sample_tree.query(cloud, workers=workers)[1]
    for start in range(0, len(cloud), 200_000):
        stop = min(start + 200_000, len(cloud))
        index = nearest[start:stop]
        delta = cloud[start:stop] - samples[index]
        normal = sample_normals[index]
        along = np.einsum("ij,ij->i", delta, normal)
        lateral = np.linalg.norm(delta - along[:, None] * normal, axis=1)
        orientation = np.abs(np.einsum("ij,ij->i", cloud_normals[start:stop], normal))
        ok = (np.abs(along) <= along_tol) & (lateral <= lateral_tol) & (orientation >= 0.5)
        supported[index[ok]] = True
    return supported


def element_vertices_faces(element, settings):
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        geometry = shape.geometry
        vertices = np.asarray(geometry.verts, dtype=float).reshape(-1, 3)
        faces = np.asarray(geometry.faces, dtype=np.int64).reshape(-1, 3)
        if len(faces):
            return vertices, faces
    except (RuntimeError, ValueError):
        pass
    part_vertices, part_faces = [], []
    offset = 0
    for rel in element.IsDecomposedBy:
        for child in rel.RelatedObjects:
            try:
                shape = ifcopenshell.geom.create_shape(settings, child)
                geometry = shape.geometry
                vertices = np.asarray(geometry.verts, dtype=float).reshape(-1, 3)
                faces = np.asarray(geometry.faces, dtype=np.int64).reshape(-1, 3)
            except (RuntimeError, ValueError):
                continue
            if not len(faces):
                continue
            part_vertices.append(vertices)
            part_faces.append(faces + offset)
            offset += len(vertices)
    if not part_vertices:
        raise ValueError("Element has no nondegenerate surface triangles")
    return np.concatenate(part_vertices), np.concatenate(part_faces)


def status_from_coverage(coverage, partial_threshold, built_threshold):
    if coverage >= built_threshold:
        return "built"
    if coverage >= partial_threshold:
        return "partial"
    return "not_built"


def run(args):
    started = time.monotonic()
    if not 0 < args.partial_threshold < args.built_threshold <= 1:
        raise ValueError("Require 0 < partial threshold < built threshold <= 1")
    if not np.isfinite([args.distance, args.cloud_scale]).all() or args.distance <= 0 or args.samples < 1 or args.cloud_scale <= 0:
        raise ValueError("Distance, sample count and cloud scale must be finite and positive")
    if args.workers != -1 and args.workers < 1:
        raise ValueError("Workers must be -1 or a positive integer")
    if not np.isfinite(args.lateral) or args.lateral <= 0:
        raise ValueError("Lateral tolerance must be finite and positive")
    if args.coverage_mode == "competitive" and not args.attributes:
        raise ValueError("Competitive coverage requires scan normals via --attributes")
    cloud = load_cloud(args.cloud) * args.cloud_scale
    cloud_normals = None
    if args.attributes:
        with np.load(args.attributes, allow_pickle=False) as archive:
            attribute_points = np.asarray(archive["points"], dtype=float)
            cloud_normals = np.asarray(archive["normals"], dtype=float)
        if attribute_points.shape != cloud.shape or not np.allclose(attribute_points[:1000] * args.cloud_scale, cloud[:1000]):
            raise ValueError("Attributes do not match the loaded cloud order/scale")
        cloud_normals = cloud_normals / np.linalg.norm(cloud_normals, axis=1)[:, None]
    transform = np.eye(4)
    if args.transform:
        transform = np.asarray(json.loads(args.transform.read_text(encoding="utf-8")), dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Transform must be a finite 4x4 matrix mapping the scaled cloud into IFC world coordinates")
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Transform must be affine")
        cloud = cloud @ transform[:3, :3].T + transform[:3, 3]
    if cloud_normals is not None:
        cloud_normals = cloud_normals @ transform[:3, :3].T
        cloud_normals = cloud_normals / np.linalg.norm(cloud_normals, axis=1)[:, None]
    tree = cKDTree(cloud)
    model = ifcopenshell.open(str(args.ifc))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    elements = {}
    for entity_type in args.types:
        for element in model.by_type(entity_type):
            elements[element.GlobalId] = element
    rows = []
    failures = []
    competitive_parts = []
    row_by_guid = {}
    for index, (guid, element) in enumerate(sorted(elements.items()), start=1):
        row = {
            "sample_id": f"{args.scan_id}:{guid}",
            "ifc_guid": guid,
            "ifc_type": element.is_a(),
            "name": element.Name or "",
            "prediction": "unobserved",
            "observed_surface_fraction": "",
            "median_surface_distance_m": "",
            "surface_area_m2": "",
            "n_surface_samples": 0,
        }
        try:
            vertices, faces = element_vertices_faces(element, settings)
            element_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{guid}".encode()).digest()[:8], "little")
            sampled, sample_normals, area = sample_surface_with_normals(vertices, faces, args.samples, np.random.default_rng(element_seed))
            distances, _ = tree.query(sampled, workers=args.workers)
            if args.coverage_mode == "competitive":
                competitive_parts.append((guid, sampled, sample_normals))
                coverage = None
            elif cloud_normals is not None:
                coverage = normal_aware_coverage(sampled, sample_normals, cloud, cloud_normals, tree, args.distance, args.lateral)
            else:
                coverage = float(np.mean(distances <= args.distance))
            row.update(
                prediction=status_from_coverage(coverage, args.partial_threshold, args.built_threshold) if coverage is not None else "unobserved",
                observed_surface_fraction=coverage if coverage is not None else "",
                median_surface_distance_m=float(np.median(distances)),
                surface_area_m2=area,
                n_surface_samples=len(sampled),
            )
        except (RuntimeError, ValueError) as error:
            failures.append({"ifc_guid": guid, "error": str(error)})
        rows.append(row)
        row_by_guid[guid] = row
        if index % 100 == 0:
            print(f"Processed {index}/{len(elements)} IFC elements", flush=True)
    if competitive_parts:
        all_samples = np.concatenate([part[1] for part in competitive_parts])
        all_normals = np.concatenate([part[2] for part in competitive_parts])
        supported = competitive_coverage(all_samples, all_normals, cloud, cloud_normals, args.distance, args.lateral, args.workers)
        offset = 0
        for guid, sampled, _ in competitive_parts:
            coverage = float(supported[offset:offset + len(sampled)].mean())
            offset += len(sampled)
            row_by_guid[guid].update(
                prediction=status_from_coverage(coverage, args.partial_threshold, args.built_threshold),
                observed_surface_fraction=coverage,
            )
        print(f"Competitive assignment over {len(all_samples)} samples and {len(cloud)} scan points", flush=True)
    if not rows:
        raise ValueError("No IFC elements matched the selected classes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "method": "area-weighted-ifc-surface-coverage-v1",
        "elapsed_seconds": time.monotonic() - started,
        "ifc_path": str(args.ifc),
        "ifc_sha256": sha256(args.ifc),
        "cloud_path": str(args.cloud),
        "cloud_sha256": sha256(args.cloud),
        "prediction_sha256": sha256(args.output),
        "n_cloud_points": len(cloud),
        "cloud_bounds_m": [cloud.min(axis=0).tolist(), cloud.max(axis=0).tolist()],
        "cloud_scale": args.cloud_scale,
        "cloud_to_ifc_transform": transform.tolist(),
        "selected_ifc_types": args.types,
        "seed": args.seed,
        "surface_samples_per_element": args.samples,
        "distance_tolerance_m": args.distance,
        "coverage_mode": (
            "competitive_nearest_face_assignment"
            if args.coverage_mode == "competitive"
            else ("normal_aware_plane_support" if args.attributes else "nearest_point_distance")
        ),
        "lateral_tolerance_m": args.lateral if args.attributes else None,
        "attributes_path": str(args.attributes) if args.attributes else None,
        "partial_threshold": args.partial_threshold,
        "built_threshold": args.built_threshold,
        "n_elements": len(rows),
        "state_counts": {state: sum(row["prediction"] == state for row in rows) for state in ("built", "partial", "not_built", "unobserved")},
        "geometry_failures": failures,
        "limitations": [
            "Surface coverage is observed evidence, not a physical percentage of construction completion.",
            "Baseline assumes registration supplied by the caller; it does not estimate alignment.",
            "Occlusion and clutter can cause false absence or false completion; this baseline has no visibility model.",
            "Heuristic thresholds require validation against independent progress labels before use.",
        ],
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return rows, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transform", type=Path)
    parser.add_argument("--cloud-scale", type=float, default=1.0)
    parser.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--distance", type=float, default=0.05)
    parser.add_argument("--lateral", type=float, default=0.15)
    parser.add_argument("--coverage-mode", choices=("plane", "competitive"), default="plane")
    parser.add_argument("--attributes", type=Path)
    parser.add_argument("--partial-threshold", type=float, default=0.1)
    parser.add_argument("--built-threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
