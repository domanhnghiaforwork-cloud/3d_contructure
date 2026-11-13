import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

CLASSES = ["built", "partial", "not_built"]


def load(labels_path, predictions_path):
    labels = {row["sample_id"]: row["label"] for row in csv.DictReader(open(labels_path, encoding="utf-8"))}
    rows = [row for row in csv.DictReader(open(predictions_path, encoding="utf-8")) if row["sample_id"] in labels]
    ids = [row["sample_id"] for row in rows]
    if set(ids) != set(labels):
        raise ValueError("Predictions must cover the frozen cohort exactly")
    coverage = np.array([float(row["observed_surface_fraction"]) if row["observed_surface_fraction"] else -1.0 for row in rows])
    truth = np.array([labels[i] for i in ids])
    return ids, coverage, truth


def predict(coverage, partial_threshold, built_threshold):
    out = np.full(len(coverage), "unobserved", dtype=object)
    ok = coverage >= 0
    out[ok & (coverage >= built_threshold)] = "built"
    out[ok & (coverage >= partial_threshold) & (coverage < built_threshold)] = "partial"
    out[ok & (coverage < partial_threshold)] = "not_built"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids, coverage, truth = load(args.labels, args.predictions)
    results = []
    for partial_threshold in np.arange(0.05, 0.51, 0.05):
        for built_threshold in np.arange(0.5, 0.96, 0.05):
            if built_threshold <= partial_threshold:
                continue
            pred = predict(coverage, partial_threshold, built_threshold)
            results.append({
                "partial_threshold": round(float(partial_threshold), 2),
                "built_threshold": round(float(built_threshold), 2),
                "accuracy": float(accuracy_score(truth, pred)),
                "macro_f1": float(f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0)),
            })
    results.sort(key=lambda r: (r["macro_f1"], r["accuracy"]), reverse=True)
    default = predict(coverage, 0.1, 0.65)
    report = {
        "purpose": "Validation-only threshold search; never run on the frozen test cohort",
        "n": len(ids),
        "default_thresholds": {"partial": 0.1, "built": 0.65, "accuracy": float(accuracy_score(truth, default)), "macro_f1": float(f1_score(truth, default, labels=CLASSES, average="macro", zero_division=0)), "confusion": confusion_matrix(truth, default, labels=CLASSES + ["unobserved"]).tolist()},
        "top10": results[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
