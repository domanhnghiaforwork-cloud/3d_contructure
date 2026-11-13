"""Apply frozen semantic-group config to baseline coverage predictions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import ifcopenshell

import semantic_groups as sg


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ifc", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--along", type=float, default=0.1)
    parser.add_argument("--lateral", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))["config"]
    with args.predictions.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    coverage = {r["ifc_guid"]: (float(r["observed_surface_fraction"]) if r["observed_surface_fraction"] else None) for r in rows}
    model = ifcopenshell.open(str(args.ifc))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    groups, bounds = sg.assign_groups(model, settings)
    hosts = sg.cavity_hosts(groups, bounds)
    cloud, cloud_normals, colors = sg.load_scan(args.attributes, args.transform)
    for row in rows:
        guid = row["ifc_guid"]
        group = groups.get(guid, "exterior")
        yellow = None
        if group == "cavity":
            yellow = sg.supporting_point_colors(guid, model, settings, cloud, cloud_normals, colors, args.transform, args.along, args.lateral)
        row["prediction"] = sg.predict_status(group, coverage[guid], config, yellow, coverage.get(hosts.get(guid)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "method": "semantic-group-thresholds-v5",
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "coverage_predictions_sha256": sha256(args.predictions),
        "prediction_sha256": sha256(args.output),
        "group_counts": {g: sum(groups.get(r["ifc_guid"]) == g for r in rows) for g in sorted(set(groups.values()))},
        "state_counts": {s: sum(r["prediction"] == s for r in rows) for s in ("built", "partial", "not_built", "unobserved")},
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata["state_counts"]))


if __name__ == "__main__":
    main()