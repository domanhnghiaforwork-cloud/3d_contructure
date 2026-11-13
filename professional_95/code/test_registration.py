import unittest

import numpy as np
from scipy.spatial import cKDTree

from register_cloud import building_frames, rigid_fit, transformed, trimmed_icp, voxel_points


class RegistrationTests(unittest.TestCase):
    def test_rigid_transform_recovery(self):
        source = np.random.default_rng(42).normal(size=(100, 3))
        angle = 0.7
        truth = np.eye(4)
        truth[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        truth[:3, 3] = [12, -3, 0.4]
        target = transformed(source, truth)
        estimated = rigid_fit(source, target)
        np.testing.assert_allclose(estimated, truth, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(estimated[:3, :3]), 1.0)

    def test_icp_preserves_initialized_scale_and_proper_gravity_flip(self):
        rng = np.random.default_rng(17)
        source = rng.uniform(-2, 2, size=(300, 3))
        initial = np.eye(4)
        initial[:3, :3] = 1.6 * np.diag([1, -1, -1])
        truth = initial.copy()
        truth[:3, 3] = [0.03, -0.02, 0.04]
        target = transformed(source, truth)
        matrix, scores = trimmed_icp(source, target, cKDTree(target), initial, 1.0, 40)
        np.testing.assert_allclose(matrix, truth, atol=1e-10)
        np.testing.assert_allclose(np.linalg.svd(matrix[:3, :3])[1], [1.6] * 3, atol=1e-12)
        self.assertGreater(np.linalg.det(matrix[:3, :3]), 0)
        self.assertLess(scores["trimmed_rmse_m"], 1e-10)

    def test_building_frames_are_proper_and_flip_gravity(self):
        rng = np.random.default_rng(3)
        points = rng.uniform(0, 10, size=(3000, 3))
        normals = np.eye(3)[rng.integers(0, 3, size=3000)]
        normals += rng.normal(0, 0.01, size=normals.shape)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        frames = building_frames(points, normals, [0, 0, 0, 10, 10, 10])
        self.assertEqual(len(frames), 4)
        for frame in frames:
            np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(np.linalg.det(frame), 1.0, places=8)
            np.testing.assert_allclose(frame[2], [0, 0, -1], atol=2e-3)
        horizontals = {tuple(np.round(frame[0], 2)) for frame in frames}
        self.assertEqual(horizontals, {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)})

    def test_voxel_representatives(self):
        points = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 0.1, 0.1]])
        np.testing.assert_array_equal(voxel_points(points, 1.0), points[[0, 2]])

    def test_icp_trims_far_outliers(self):
        rng = np.random.default_rng(7)
        target = rng.uniform(-2, 2, size=(400, 3))
        source = np.vstack([target - [0.05, -0.03, 0.02], rng.uniform(10, 15, size=(100, 3))])
        matrix, scores = trimmed_icp(source, target, cKDTree(target), np.eye(4), 0.8, 40)
        np.testing.assert_allclose(matrix[:3, 3], [0.05, -0.03, 0.02], atol=1e-8)
        self.assertLess(scores["trimmed_rmse_m"], 1e-8)


if __name__ == "__main__":
    unittest.main()
