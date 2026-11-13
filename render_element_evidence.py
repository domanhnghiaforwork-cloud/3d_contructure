import argparse
import csv
import json
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import matplotlib
import numpy as np
from matplotlib.collections import PolyCollection

from progress_baseline import element_vertices_faces, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def storey_map(model):
    mapping = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        for element in rel.RelatedElements:
            mapping[element.GlobalId] = rel.RelatingStructure.Name or ""
    return mapping


def render(args):
    model = ifcopenshell.open(str(args.ifc))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    storeys = storey_map(model)
    matrix = np.asarray(json.loads(args.transform.read_text(encoding="utf-8")), dtype=float)
    with np.load(args.attributes, allow_pickle=False) as archive:
        points = archive["points"] @ matrix[:3, :3].T + matrix[:3, 3]
        colors = archive["colors"]
    guids = [line.strip() for line in args.guids.read_text(encoding="utf-8").splitlines() if line.strip()]
    elements = {element.GlobalId: element for element in model.by_type("IfcProduct") if element.GlobalId in set(guids)}
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for guid in guids:
        element = elements[guid]
        vertices, faces = element_vertices_faces(element, settings)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        margin = 0.5
        keep = np.all((points >= low - margin) & (points <= high + margin), axis=1)
        local, rgb = points[keep], colors[keep]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for axis, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2))):
            axis.scatter(local[:, a], local[:, b], c=rgb, s=1.2)
            silhouette = PolyCollection(vertices[faces][:, :, [a, b]], facecolors="#cc433d", edgecolors="none", alpha=0.25)
            axis.add_collection(silhouette)
            axis.set_xlim(low[a] - margin, high[a] + margin)
            axis.set_ylim(low[b] - margin, high[b] + margin)
            axis.set_aspect("equal")
            axis.set_xlabel("XYZ"[a] + " (m)")
            axis.set_ylabel("XYZ"[b] + " (m)")
            axis.grid(alpha=0.2)
        fig.suptitle(f"{guid} | {element.is_a()} | {element.Name or ''} | storey: {storeys.get(guid, '?')}\nRed silhouette = designed IFC geometry; dots = registered scan RGB. No classifier output shown.", fontsize=9)
        fig.tight_layout()
        path = args.output / f"{guid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        index.append({"ifc_guid": guid, "ifc_type": element.is_a(), "name": element.Name or "", "storey": storeys.get(guid, ""), "bounds_m": [low.tolist(), high.tolist()], "n_scan_points_in_box": int(keep.sum()), "evidence_png": str(path)})
        print(f"Rendered {guid} ({int(keep.sum())} scan points)", flush=True)
    (args.output / "evidence_index.json").write_text(json.dumps({"ifc_sha256": sha256(args.ifc), "transform_sha256": sha256(args.transform), "attributes_sha256": sha256(args.attributes), "purpose": "Visual reference-labeling evidence only; contains no classifier predictions or coverage values.", "elements": index}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--guids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args())
