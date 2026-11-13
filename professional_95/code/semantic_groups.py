"""Semantic visibility groups for progress classification.

Groups encode construction/observability domain knowledge, not label information:
cavity insulation is occluded by design leafs, interior faces are only partially
observable from an exterior drone scan, roof coverings are fully observable from
above, and steel sections are slender exposed members.
"""

import hashlib
import json

import numpy as np
import ifcopenshell
from scipy.spatial import cKDTree

import progress_baseline as pb

TYPES = ["IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof", "IfcBeam", "IfcColumn", "IfcCovering"]
INTERIOR_KEYS = ("binnen", "lift", "separatie", "dekvloer", "vloer_v0", "plafond", "tegelwerk")
ROOF_KEYS = ("dakpan", "tegels", "zink", "waterslag")


def element_bounds(element, settings):
    vertices, _ = pb.element_vertices_faces(element, settings)
    return vertices.min(axis=0), vertices.max(axis=0)


def assign_groups(model, settings):
    groups = {}
    bounds = {}
    for entity_type in TYPES:
        for element in model.by_type(entity_type):
            guid = element.GlobalId
            name = (element.Name or "").lower()
            try:
                bounds[guid] = element_bounds(element, settings)
            except (RuntimeError, ValueError):
                bounds[guid] = None
            if "isolatie" in name:
                group = "cavity"
            elif element.is_a() in ("IfcBeam", "IfcColumn"):
                group = "struct"
            elif element.is_a() == "IfcRoof" or (element.is_a() == "IfcCovering" and any(k in name for k in ROOF_KEYS)):
                group = "roof"
            elif any(k in name for k in INTERIOR_KEYS):
                group = "interior"
            elif element.is_a() == "IfcSlab":
                group = "top"
            else:
                group = "exterior"
            groups[guid] = group
    return groups, bounds


def cavity_hosts(groups, bounds):
    hosts = {}
    walls = [g for g, grp in groups.items() if grp == "exterior" and bounds[g] is not None]
    for guid, group in groups.items():
        if group != "cavity" or bounds[guid] is None:
            continue
        lo, hi = bounds[guid]
        best, best_overlap = None, 0.0
        for wall in walls:
            wlo, whi = bounds[wall]
            overlap = np.minimum(hi, whi) - np.maximum(lo, wlo)
            if (overlap[:2] > 0.05).all() and overlap[2] > 0.5:
                volume = float(np.prod(np.maximum(overlap, 0)))
                if volume > best_overlap:
                    best, best_overlap = wall, volume
        hosts[guid] = best
    return hosts


def supporting_point_colors(guid, model, settings, cloud, cloud_normals, colors, transform, along_tol, lateral_tol, samples=2000, seed=42):
    element = model.by_guid(guid)
    element_seed = int.from_bytes(hashlib.sha256(f"{seed}:{guid}".encode()).digest()[:8], "little")
    vertices, faces = pb.element_vertices_faces(element, settings)
    sampled, normals, _ = pb.sample_surface_with_normals(vertices, faces, samples, np.random.default_rng(element_seed))
    tree = cKDTree(cloud)
    picked = []
    for index, neighbors in enumerate(tree.query_ball_point(sampled, r=along_tol + lateral_tol)):
        if not neighbors:
            continue
        delta = cloud[neighbors] - sampled[index]
        along = delta @ normals[index]
        lateral = np.linalg.norm(delta - along[:, None] * normals[index], axis=1)
        orientation = np.abs(cloud_normals[neighbors] @ normals[index])
        ok = (np.abs(along) <= along_tol) & (lateral <= lateral_tol) & (orientation >= 0.5)
        picked.extend(np.asarray(neighbors)[ok].tolist())
    if not picked:
        return None
    rgb = colors[np.unique(picked)].astype(float)
    yellow = (rgb[:, 0] > 140) & (rgb[:, 1] > 110) & (rgb[:, 2] < 110) & ((rgb[:, 0] - rgb[:, 2]) > 50)
    return float(yellow.mean())


def predict_status(group, coverage, config, yellow=None, host_coverage=None):
    if coverage is None:
        return "unobserved"
    if group == "cavity":
        cavity = config["cavity"]
        if yellow is not None and yellow >= cavity["yellow_fraction"]:
            return "built"
        if host_coverage is not None and host_coverage >= cavity["host_built_threshold"]:
            return "built"
        return "not_built"
    thresholds = config["groups"][group]
    if coverage >= thresholds["built"]:
        return "built"
    if coverage >= thresholds["partial"]:
        return "partial"
    return "not_built"


def load_scan(attributes_path, transform_path):
    with np.load(attributes_path, allow_pickle=False) as archive:
        points = np.asarray(archive["points"], dtype=float)
        normals = np.asarray(archive["normals"], dtype=float)
        colors = np.stack([archive["colors"][:, i] for i in range(3)], axis=1) if "colors" in archive else None
    transform = np.asarray(json.loads(transform_path.read_text(encoding="utf-8")), dtype=float)
    points = points @ transform[:3, :3].T + transform[:3, 3]
    normals = normals @ transform[:3, :3].T
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    return points, normals, colors