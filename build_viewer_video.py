"""Đóng gói cloud DỰNG LẠI TỪ VIDEO + mẫu bề mặt BIM theo trạng thái dự đoán
into viewer_data.js (cùng định dạng build_viewer_3d.py để viewer_3d.html chạy).

Cloud đầu vào ở khung ref (alignment_ref/recon_attributes.npz) → áp G3 ref→IFC.
predictions/summary lấy từ eval của pipeline video. Không dùng CDN/URL ngoài.
"""
import argparse
import base64
import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import ifcopenshell

import progress_baseline as pb

ROOT = Path(__file__).resolve().parent
TYPES = ["IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof", "IfcBeam", "IfcColumn", "IfcCovering"]
STATUS = ["built", "partial", "not_built", "unobserved"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", type=Path, required=True, help="recon_attributes.npz (khung ref)")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-points", type=int, default=120_000)
    args = ap.parse_args()

    attributes = np.load(args.cloud)
    points = attributes["points"].astype(np.float64)
    colors = attributes["colors"]
    colors = np.clip(colors * 255 if colors.dtype.kind == "f" else colors, 0, 255).astype(np.uint8)
    transform = np.asarray(json.loads((ROOT / "artifacts/week26_registration_v3/candidate_transform.json").read_text()))
    points = points @ transform[:3, :3].T + transform[:3, 3]

    rng = np.random.default_rng(7)
    keys = np.floor(points / 0.12).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    if len(first) > args.max_points:
        first = rng.choice(first, args.max_points, replace=False)
    scan_points = points[first].astype(np.float32)
    scan_colors = colors[first]
    scan_blob = b"".join(struct.pack("<3f3B", *p, *c) for p, c in zip(scan_points, scan_colors))

    predictions = {r["ifc_guid"]: r for r in csv.DictReader(args.predictions.open(encoding="utf-8-sig"))}
    labels = {}
    for split, name in (("validation", "validation_labels.csv"), ("test", "test_labels_frozen.csv")):
        for row in csv.DictReader((ROOT / "artifacts/cohort_v1" / name).open(encoding="utf-8-sig")):
            labels[row["sample_id"].replace("week26:", "")] = (row["label"], split)

    model = ifcopenshell.open(str(ROOT / "data/schependomlaan/raw/model.ifc"))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    storeys = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        name = rel.RelatingStructure.Name or rel.RelatingStructure.is_a()
        for element in rel.RelatedElements:
            storeys[element.GlobalId] = name
    storey_names = sorted(set(storeys.values()))
    storey_index = {name: i for i, name in enumerate(storey_names)}

    elements = []
    chunks = []
    seen = set()
    for type_index, entity_type in enumerate(TYPES):
        for element in sorted(model.by_type(entity_type), key=lambda e: e.GlobalId):
            guid = element.GlobalId
            if guid in seen:
                continue
            seen.add(guid)
            type_index = TYPES.index(element.is_a()) if element.is_a() in TYPES else type_index
            row = predictions.get(guid)
            if row is None:
                continue
            try:
                vertices, faces = pb.element_vertices_faces(element, settings)
            except (RuntimeError, ValueError):
                continue
            seed = int.from_bytes(hashlib.sha256(f"42:{guid}".encode()).digest()[:8], "little")
            samples, _, area = pb.sample_surface_with_normals(vertices, faces, 240, np.random.default_rng(seed))
            status = STATUS.index(row["prediction"])
            coverage = float(row["observed_surface_fraction"]) if row["observed_surface_fraction"] else 0.0
            storey = storey_index.get(storeys.get(guid, ""), 0)
            element_index = len(elements)
            label, split = labels.get(guid, (None, None))
            elements.append({"g": guid, "n": element.Name or "", "t": type_index, "s": storey,
                             "c": round(coverage, 4), "p": status, "a": round(float(area), 2),
                             "l": label, "sp": split})
            for point in samples.astype(np.float32):
                chunks.append(struct.pack("<3fBBBBH", *point, status, int(round(coverage * 255)), storey, type_index, element_index))
    bim_blob = b"".join(chunks)

    summary = json.loads(args.summary.read_text())
    stats = {status: sum(e["p"] == i for e in elements) for i, status in enumerate(STATUS)}
    data = {
        "scanB64": base64.b64encode(scan_blob).decode(),
        "bimB64": base64.b64encode(bim_blob).decode(),
        "scanCount": len(scan_points), "bimCount": len(chunks),
        "elements": elements, "storeys": storey_names, "types": TYPES,
        "status": STATUS, "stats": stats,
        "testAccuracy": round(summary["test"]["accuracy"], 4),
        "testMacroF1": round(summary["test"]["macro_f1"], 4),
        "scanBounds": [scan_points.min(axis=0).tolist(), scan_points.max(axis=0).tolist()],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("window.VIEWER_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("scan", len(scan_points), "| bim", len(chunks), "| elements", len(elements),
          "| testAcc", data["testAccuracy"], "| bytes", args.out.stat().st_size, flush=True)


if __name__ == "__main__":
    main()