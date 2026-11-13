import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import ifcopenshell.api
import ifcopenshell.geom
import numpy as np

from progress_baseline import load_cloud, run, sample_surface, status_from_coverage


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_surface_samples_stay_inside_triangle(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        points, area = sample_surface(vertices, np.array([[0, 1, 2]]), 1000, np.random.default_rng(42))
        self.assertEqual(area, 0.5)
        self.assertTrue((points >= 0).all())
        self.assertTrue((points[:, :2].sum(axis=1) <= 1).all())
        self.assertTrue((points[:, 2] == 0).all())

    def test_normal_aware_coverage_rejects_parallel_offset_face(self):
        from scipy.spatial import cKDTree

        from progress_baseline import normal_aware_coverage
        samples = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        cloud = np.array([[0.0, 0.0, 0.15], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
        cloud_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        coverage = normal_aware_coverage(samples, normals, cloud, cloud_normals, cKDTree(cloud), 0.05, 0.15)
        self.assertEqual(coverage, 0.5)

    def test_competitive_coverage_assigns_point_to_nearest_face_only(self):
        from progress_baseline import competitive_coverage, normal_aware_coverage
        from scipy.spatial import cKDTree

        samples = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.08], [1.0, 0.0, 0.08]])
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        cloud = np.array([[0.0, 0.0, 0.08], [1.0, 0.0, 0.08]])
        cloud_normals = np.tile([0.0, 0.0, 1.0], (2, 1))
        supported = competitive_coverage(samples, normals, cloud, cloud_normals, 0.1, 0.15, 1)
        self.assertEqual(supported.tolist(), [False, False, True, True])
        plane = normal_aware_coverage(samples[:2], normals[:2], cloud, cloud_normals, cKDTree(cloud), 0.1, 0.15)
        self.assertEqual(plane, 1.0)

    def test_threshold_boundaries(self):
        self.assertEqual(status_from_coverage(0.0, 0.1, 0.65), "not_built")
        self.assertEqual(status_from_coverage(0.1, 0.1, 0.65), "partial")
        self.assertEqual(status_from_coverage(0.65, 0.1, 0.65), "built")

    def test_cloud_nonfinite_rejected(self):
        path = self.directory / "cloud.npy"
        np.save(path, np.array([[0.0, float("nan"), 1.0]]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_cloud(path)

    def test_degenerate_surface_rejected(self):
        with self.assertRaisesRegex(ValueError, "nondegenerate"):
            sample_surface(np.zeros((3, 3)), np.array([[0, 1, 2]]), 10, np.random.default_rng(42))

    def test_nonfinite_distance_is_rejected_before_inference(self):
        args = argparse.Namespace(partial_threshold=0.1, built_threshold=0.65, distance=float("nan"), cloud_scale=1.0, samples=100)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            run(args)

    def test_synthetic_ifc_smoke_and_geometry_failure_retained(self):
        model = ifcopenshell.api.run("project.create_file")
        ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="Synthetic unit-test fixture")
        ifcopenshell.api.run("unit.assign_unit", model)
        context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
        body = ifcopenshell.api.run("context.add_context", model, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=context)
        wall = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWall", name="Fixture wall")
        representation = ifcopenshell.api.run("geometry.add_wall_representation", model, context=body, length=1.0, height=1.0, thickness=0.2)
        ifcopenshell.api.run("geometry.assign_representation", model, product=wall, representation=representation)
        ifcopenshell.api.run("geometry.edit_object_placement", model, product=wall, matrix=np.eye(4))
        empty = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWall", name="No geometry fixture")
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        shape = ifcopenshell.geom.create_shape(settings, wall)
        vertices = np.asarray(shape.geometry.verts).reshape(-1, 3)
        faces = np.asarray(shape.geometry.faces).reshape(-1, 3)
        points, _ = sample_surface(vertices, faces, 10000, np.random.default_rng(7))
        ifc_path = self.directory / "synthetic.ifc"
        cloud_path = self.directory / "synthetic.npy"
        output = self.directory / "predictions.csv"
        model.write(str(ifc_path))
        np.save(cloud_path, points)
        args = argparse.Namespace(
            ifc=ifc_path, cloud=cloud_path, output=output, scan_id="unit-test-only",
            transform=None, cloud_scale=1.0, types=["IfcWall"], samples=500,
            distance=0.05, lateral=0.15, attributes=None, coverage_mode="plane", partial_threshold=0.1, built_threshold=0.65, seed=42, workers=1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            rows, metadata = run(args)
        by_guid = {row["ifc_guid"]: row for row in rows}
        self.assertEqual(by_guid[wall.GlobalId]["prediction"], "built")
        self.assertEqual(by_guid[empty.GlobalId]["prediction"], "unobserved")
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(metadata["geometry_failures"]), 1)
        self.assertTrue(output.exists())
        saved = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertEqual(saved["n_elements"], 2)
        self.assertEqual(saved["cloud_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
