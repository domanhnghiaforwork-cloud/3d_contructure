import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import numpy as np

from progress_baseline import DEFAULT_TYPES, sha256


def material_names(element):
    names = []

    def walk(material):
        if material is None:
            return
        if material.is_a("IfcMaterial"):
            names.append(material.Name or "")
        elif material.is_a("IfcMaterialList"):
            for item in material.Materials:
                walk(item)
        elif material.is_a("IfcMaterialLayerSetUsage"):
            walk(material.ForLayerSet)
        elif material.is_a("IfcMaterialLayerSet"):
            for layer in material.MaterialLayers:
                walk(layer.Material)
        elif material.is_a("IfcMaterialConstituentSet"):
            for constituent in material.MaterialConstituents:
                walk(constituent.Material)
        elif material.is_a("IfcMaterialProfileSetUsage"):
            walk(material.ForProfileSet)
        elif material.is_a("IfcMaterialProfileSet"):
            for profile in material.MaterialProfiles:
                walk(profile.Material)

    for association in element.HasAssociations:
        if association.is_a("IfcRelAssociatesMaterial"):
            walk(association.RelatingMaterial)
    return names


def body_children(element):
    children = []
    for rel in element.IsDecomposedBy:
        children.extend(rel.RelatedObjects)
    return children


def has_body(element):
    if not element.Representation:
        return False
    return any(rep.RepresentationIdentifier in ("Body", "Facetation") for rep in element.Representation.Representations)


def define(args):
    model = ifcopenshell.open(str(args.ifc))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    types = list(DEFAULT_TYPES) + ["IfcCovering"]
    elements = {element.GlobalId: element for entity_type in types for element in model.by_type(entity_type)}
    rows = []
    reasons = Counter()
    for guid, element in sorted(elements.items()):
        names = material_names(element)
        air_only = bool(names) and all("Lucht" in name for name in names)
        children = body_children(element)
        row = {"ifc_guid": guid, "ifc_type": element.is_a(), "name": element.Name or "", "materials": ";".join(names), "child_guids": ";".join(child.GlobalId for child in children), "eligibility": "", "reason": ""}
        if air_only:
            row.update(eligibility="excluded", reason="nonphysical_air_only")
        elif not has_body(element) and children:
            row.update(eligibility="eligible", reason="decomposed_parent_use_children")
        else:
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
                vertices = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
                faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
                triangles = vertices[faces]
                area = float(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2)
                if area <= 1e-9:
                    raise ValueError("zero area")
                row.update(eligibility="eligible", reason="solid_geometry", area_m2=area)
            except (RuntimeError, ValueError):
                row.update(eligibility="excluded", reason="empty_after_opening_subtraction")
        reasons[row["reason"]] += 1
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cohort_eligibility.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ifc_guid", "ifc_type", "name", "materials", "child_guids", "eligibility", "reason", "area_m2"])
        writer.writeheader()
        writer.writerows(rows)
    report = {"ifc_path": str(args.ifc), "ifc_sha256": sha256(args.ifc), "types": types, "n_elements": len(rows), "reasons": dict(reasons), "n_eligible": sum(row["eligibility"] == "eligible" for row in rows), "rule": "Physical-scope eligibility decided from IFC semantics only (materials, openings, decomposition), before any classifier prediction or label inspection.", "caveat": "Eligibility is not a progress label and not yet a frozen test cohort; labeling and visibility screening come next."}
    (args.output / "cohort_eligibility.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    define(parser.parse_args())
