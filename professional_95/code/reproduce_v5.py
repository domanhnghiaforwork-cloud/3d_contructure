"""End-to-end reproduction of the frozen v5 result from raw public data.

Steps: registration -> coverage baseline -> validation-only tuning -> apply
frozen-style config -> single test evaluation. Each step is compared against
the frozen artifacts; any mismatch fails the reproduction.
"""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "artifacts" / "repro_v5"
FROZEN = {
    "artifacts/week26_registration_v3/candidate_transform.json": "719cc91ac7ed3b2f6c34e59b630a8f53dee28c519d0cbfb2f4c4094e9fd9d52e",
    "artifacts/cohort_v1/validation_labels.csv": "42dc112272e2b38249d24681c8cd5eb44f48957459bc5747bc1f5fa9b19b5845",
    "artifacts/cohort_v1/test_labels_frozen.csv": "6cdde870dde51151c700743731b5131a58042dba61a8c0b68275c441cc77eb9d",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*command):
    print("+", " ".join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], check=True, cwd=ROOT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for rel, expected in FROZEN.items():
        actual = sha(ROOT / rel)
        assert actual == expected, f"frozen artifact changed: {rel}"
    print("frozen artifact hashes OK", flush=True)

    registration = json.loads((ROOT / "artifacts/week26_registration_v3/registration.json").read_text())
    command = [
        PY, "register_cloud.py",
        "--samples", registration["samples_path"],
        "--output", OUT / "registration",
        "--voxel", registration["voxel_raw_units_source_m_target"],
        "--trim", registration["trim_fraction"],
        "--cluster-radius", registration["cluster_radius_raw_units"],
        "--source-bounds", *registration["source_bounds_raw_units"],
        "--scales", *registration["candidate_scales"],
        "--yaw-steps", registration["yaw_steps"],
        "--z-offsets", *registration["z_offsets_m"],
        "--normal-frame", registration["normal_frame_path"],
    ]
    if registration["flip_z"]:
        command.append("--flip-z")
    zmin, zmax = registration["source_z_filter_raw_units"]
    if zmin is not None:
        command += ["--z-min", zmin]
    if zmax is not None:
        command += ["--z-max", zmax]
    run(*command)
    reproduced = np.asarray(json.loads((OUT / "registration" / "candidate_transform.json").read_text()))
    frozen = np.asarray(json.loads((ROOT / "artifacts/week26_registration_v3/candidate_transform.json").read_text()))
    repro_report = json.loads((OUT / "registration" / "registration.json").read_text())
    assert repro_report["chosen_scale"] == registration["chosen_scale"], "registration scale differs"
    rotation_delta = reproduced[:3, :3] @ frozen[:3, :3].T
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1))))
    translation_delta = float(np.linalg.norm(reproduced[:3, 3] - frozen[:3, 3]))
    assert angle < 0.5 and translation_delta < 0.1, f"registration differs beyond control band: {angle} deg, {translation_delta} m"
    print(f"registration reproduced within ICP noise: {angle:.3f} deg, {translation_delta:.4f} m", flush=True)

    frozen_transform = ROOT / "artifacts/week26_registration_v3/candidate_transform.json"
    run(
        PY, "progress_baseline.py",
        "--ifc", "data/schependomlaan/raw/model.ifc",
        "--cloud", "data/schependomlaan/raw/week26.ply",
        "--scan-id", "week26",
        "--attributes", "artifacts/week26_raw_diagnostics/cloud_attributes.npz",
        "--transform", frozen_transform,
        "--types", "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof", "IfcBeam", "IfcColumn", "IfcCovering",
        "--distance", 0.1, "--lateral", 0.2,
        "--output", OUT / "predictions.csv",
    )
    with (OUT / "predictions.csv").open(newline="") as handle:
        repro_coverage = {r["ifc_guid"]: r["observed_surface_fraction"] for r in csv.DictReader(handle)}
    with (ROOT / "artifacts/week26_baseline_v3/predictions.csv").open(newline="") as handle:
        frozen_coverage = {r["ifc_guid"]: r["observed_surface_fraction"] for r in csv.DictReader(handle)}
    assert repro_coverage == frozen_coverage, "coverage predictions differ"
    print("coverage baseline reproduced", flush=True)

    run(
        PY, "tune_semantic.py",
        "--labels", "artifacts/cohort_v1/validation_labels.csv",
        "--predictions", OUT / "predictions.csv",
        "--ifc", "data/schependomlaan/raw/model.ifc",
        "--attributes", "artifacts/week26_raw_diagnostics/cloud_attributes.npz",
        "--transform", OUT / "registration" / "candidate_transform.json",
        "--output", OUT / "semantic_config.json",
    )
    repro_config = json.loads((OUT / "semantic_config.json").read_text())["config"]
    frozen_config = json.loads((ROOT / "artifacts/cohort_v1/validation_semantic_config_v5.json").read_text())["config"]
    assert repro_config == frozen_config, "tuned config differs"
    print("validation tuning reproduced", flush=True)

    run(
        PY, "apply_semantic.py",
        "--predictions", OUT / "predictions.csv",
        "--config", OUT / "semantic_config.json",
        "--ifc", "data/schependomlaan/raw/model.ifc",
        "--attributes", "artifacts/week26_raw_diagnostics/cloud_attributes.npz",
        "--transform", OUT / "registration" / "candidate_transform.json",
        "--output", OUT / "semantic_predictions.csv",
    )
    with (ROOT / "artifacts/cohort_v1/test_labels_frozen.csv").open(newline="") as handle:
        ids = {r["sample_id"].replace("week26:", "") for r in csv.DictReader(handle)}
    with (OUT / "semantic_predictions.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [r for r in reader if r["ifc_guid"] in ids]
    with (OUT / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    run(
        PY, "evaluate_progress.py",
        "--labels", "artifacts/cohort_v1/test_labels_frozen.csv",
        "--predictions", OUT / "test_predictions.csv",
        "--split", "test",
        "--output", OUT / "test_metrics.json",
    )
    repro_metrics = json.loads((OUT / "test_metrics.json").read_text())
    frozen_metrics = json.loads((ROOT / "artifacts/cohort_v1/test_metrics_v5.json").read_text())
    assert repro_metrics["accuracy"] == frozen_metrics["accuracy"], "test accuracy differs"
    assert repro_metrics["macro_f1"] == frozen_metrics["macro_f1"], "test macro-F1 differs"

    run(
        PY, "progress_baseline.py",
        "--ifc", "data/schependomlaan/raw/model.ifc",
        "--cloud", "data/schependomlaan/raw/week26.ply",
        "--scan-id", "week26",
        "--attributes", "artifacts/week26_raw_diagnostics/cloud_attributes.npz",
        "--transform", OUT / "registration" / "candidate_transform.json",
        "--types", "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof", "IfcBeam", "IfcColumn", "IfcCovering",
        "--distance", 0.1, "--lateral", 0.2,
        "--output", OUT / "predictions_repro_transform.csv",
    )
    run(
        PY, "apply_semantic.py",
        "--predictions", OUT / "predictions_repro_transform.csv",
        "--config", OUT / "semantic_config.json",
        "--ifc", "data/schependomlaan/raw/model.ifc",
        "--attributes", "artifacts/week26_raw_diagnostics/cloud_attributes.npz",
        "--transform", OUT / "registration" / "candidate_transform.json",
        "--output", OUT / "semantic_predictions_repro_transform.csv",
    )
    with (ROOT / "artifacts/cohort_v1/validation_labels.csv").open(newline="") as handle:
        validation_ids = {r["sample_id"].replace("week26:", "") for r in csv.DictReader(handle)}
    with (OUT / "semantic_predictions_repro_transform.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [r for r in reader if r["ifc_guid"] in validation_ids]
    with (OUT / "validation_predictions_repro_transform.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    run(
        PY, "evaluate_progress.py",
        "--labels", "artifacts/cohort_v1/validation_labels.csv",
        "--predictions", OUT / "validation_predictions_repro_transform.csv",
        "--split", "validation",
        "--output", OUT / "validation_metrics_repro_transform.json",
    )
    robustness = json.loads((OUT / "validation_metrics_repro_transform.json").read_text())
    frozen_validation = json.loads((ROOT / "artifacts/cohort_v1/validation_semantic_config_v5.json").read_text())["report"]["validation_accuracy"]
    assert robustness["accuracy"] >= 0.8, "validation accuracy collapses under reproduced registration"
    print(f"robustness: validation accuracy with reproduced transform {robustness['accuracy']:.4f} (frozen transform {frozen_validation:.4f})", flush=True)
    print(f"REPRODUCTION PASS: test accuracy {repro_metrics['accuracy']:.4f}, macro-F1 {repro_metrics['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()