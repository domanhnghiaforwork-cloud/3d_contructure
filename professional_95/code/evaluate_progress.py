import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix


STATES = ("built", "partial", "not_built")
UNKNOWN = "unobserved"


def read_rows(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    seen = set()
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in seen:
            raise ValueError(f"{path}: empty or duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        if any(not row.get(field) for field in required):
            raise ValueError(f"{path}: empty required field for {sample_id}")
    return rows


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson_interval(correct, count):
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def evaluate(labels_path, predictions_path, split):
    labels = read_rows(labels_path, ("sample_id", "label", "split", "label_source"))
    predictions = read_rows(predictions_path, ("sample_id", "prediction"))
    for row in labels:
        if row["label"] not in STATES:
            raise ValueError(f"Invalid reference label: {row['label']}")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")
    selected = [row for row in labels if row["split"] == split]
    if not selected:
        raise ValueError(f"No reference labels for split {split}")
    expected = {row["sample_id"] for row in selected}
    actual = {row["sample_id"] for row in predictions}
    if actual != expected:
        raise ValueError(
            f"Prediction IDs must exactly match the frozen {split} cohort: "
            f"{len(expected - actual)} missing, {len(actual - expected)} extra"
        )
    by_id = {row["sample_id"]: row["prediction"] for row in predictions}
    if any(state not in (*STATES, UNKNOWN) for state in by_id.values()):
        raise ValueError("Predictions must be built, partial, not_built or unobserved")
    truth = [row["label"] for row in selected]
    predicted = [by_id[row["sample_id"]] for row in selected]
    correct = sum(reference == prediction for reference, prediction in zip(truth, predicted))
    count = len(truth)
    report = classification_report(truth, predicted, labels=list(STATES), output_dict=True, zero_division=0)
    class_support = {state: truth.count(state) for state in STATES}
    all_classes_present = all(class_support.values())
    return {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "reference_sha256": file_sha256(labels_path),
        "predictions_sha256": file_sha256(predictions_path),
        "label_sources": sorted({row["label_source"] for row in selected}),
        "n_samples": count,
        "n_correct": correct,
        "accuracy": correct / count,
        "accuracy_wilson_95_interval": wilson_interval(correct, count),
        "interval_caveat": "Binomial interval; does not account for correlated elements within a site or scan.",
        "macro_f1": report["macro avg"]["f1-score"],
        "majority_class": max(class_support, key=class_support.get),
        "majority_baseline_accuracy": max(class_support.values()) / count,
        "beats_majority_baseline": correct > max(class_support.values()),
        "class_support": class_support,
        "per_class": {state: report[state] for state in STATES},
        "confusion_labels": [*STATES, UNKNOWN],
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[*STATES, UNKNOWN]).tolist(),
        "n_unobserved": predicted.count(UNKNOWN),
        "all_three_reference_classes_present": all_classes_present,
        "meets_numeric_threshold": correct / count >= 0.8 and all_classes_present,
        "qualification_caveat": "Numeric threshold alone is not project completion; independent real-world labels and a locked test protocol must be verified separately.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.labels, args.predictions, args.split)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
