"""Validation-only tuning of per-semantic-group thresholds.

Groups partition the elements, so each group's thresholds are optimized on its
own validation subset (accuracy, tie-break macro-F1 within the subset). The
frozen test cohort is never read by this script.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import ifcopenshell

import semantic_groups as sg


def read_coverage(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {r["ifc_guid"]: (float(r["observed_surface_fraction"]) if r["observed_surface_fraction"] else None) for r in csv.DictReader(handle)}


def macro_f1(labels, predictions):
    classes = sorted(set(labels) | set(predictions))
    scores = []
    for cls in classes:
        tp = sum(l == cls and p == cls for l, p in zip(labels, predictions))
        fp = sum(l != cls and p == cls for l, p in zip(labels, predictions))
        fn = sum(l == cls and p != cls for l, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ifc", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--along", type=float, default=0.1)
    parser.add_argument("--lateral", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.labels.open(newline="", encoding="utf-8-sig") as handle:
        labels = {r["sample_id"].replace("week26:", ""): r["label"] for r in csv.DictReader(handle)}
    coverage = read_coverage(args.predictions)
    model = ifcopenshell.open(str(args.ifc))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    groups, bounds = sg.assign_groups(model, settings)
    hosts = sg.cavity_hosts(groups, bounds)
    cloud, cloud_normals, colors = sg.load_scan(args.attributes, args.transform)
    yellow = {}
    for guid in labels:
        if groups.get(guid) == "cavity":
            yellow[guid] = sg.supporting_point_colors(guid, model, settings, cloud, cloud_normals, colors, args.transform, args.along, args.lateral)

    def predict(guid, config):
        return sg.predict_status(groups[guid], coverage[guid], config, yellow.get(guid), coverage.get(hosts.get(guid)))

    config = {"groups": {}, "cavity": {"host_built_threshold": 0.3, "yellow_fraction": 0.5}}
    report = {"purpose": "Validation-only per-group threshold tuning; test cohort never read", "groups": {}}
    for group in sorted({groups[g] for g in labels}):
        subset = [g for g in labels if groups[g] == group]
        truth = [labels[g] for g in subset]
        candidates = []
        if group == "cavity":
            for host_threshold in (0.2, 0.3, 0.4, 0.5):
                candidates.append({"host_built_threshold": host_threshold, "yellow_fraction": 0.5})
        elif group == "interior":
            for threshold in np.arange(0.05, 0.35, 0.05):
                candidates.append({"partial": round(float(threshold), 2), "built": round(float(threshold), 2)})
        else:
            for partial in np.arange(0.05, 0.45, 0.05):
                for built in np.arange(0.4, 0.85, 0.05):
                    if partial < built:
                        candidates.append({"partial": round(float(partial), 2), "built": round(float(built), 2)})
        best, best_score = None, None
        for candidate in candidates:
            trial = json.loads(json.dumps(config))
            if group == "cavity":
                trial["cavity"] = candidate
            else:
                trial["groups"][group] = candidate
            predictions = [predict(g, trial) for g in subset]
            accuracy = float(np.mean([p == t for p, t in zip(predictions, truth)]))
            score = (accuracy, macro_f1(truth, predictions))
            if best_score is None or score > best_score:
                best, best_score = candidate, score
        if group == "cavity":
            config["cavity"] = best
        else:
            config["groups"][group] = best
        report["groups"][group] = {
            "n_validation": len(subset),
            "chosen": best,
            "subset_accuracy": best_score[0],
            "subset_macro_f1": best_score[1],
        }
    predictions = [predict(g, config) for g in labels]
    truth = [labels[g] for g in labels]
    report["validation_accuracy"] = float(np.mean([p == t for p, t in zip(predictions, truth)]))
    report["validation_macro_f1"] = macro_f1(truth, predictions)
    report["confusion"] = {
        f"{t}->{p}": sum(x == t and y == p for x, y in zip(truth, predictions))
        for t in ("built", "partial", "not_built")
        for p in ("built", "partial", "not_built", "unobserved")
        if sum(x == t and y == p for x, y in zip(truth, predictions))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": config, "report": report}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()