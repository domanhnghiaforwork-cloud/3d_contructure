import unittest

from semantic_groups import predict_status

CONFIG = {
    "groups": {
        "exterior": {"partial": 0.15, "built": 0.4},
        "interior": {"partial": 0.15, "built": 0.15},
        "roof": {"partial": 0.3, "built": 0.4},
    },
    "cavity": {"host_built_threshold": 0.3, "yellow_fraction": 0.5},
}


class SemanticRuleTests(unittest.TestCase):
    def test_exterior_thresholds(self):
        self.assertEqual(predict_status("exterior", 0.07, CONFIG), "not_built")
        self.assertEqual(predict_status("exterior", 0.2, CONFIG), "partial")
        self.assertEqual(predict_status("exterior", 0.56, CONFIG), "built")

    def test_interior_is_binary_at_single_threshold(self):
        self.assertEqual(predict_status("interior", 0.147, CONFIG), "not_built")
        self.assertEqual(predict_status("interior", 0.158, CONFIG), "built")

    def test_roof_rejects_beam_leakage_below_partial(self):
        self.assertEqual(predict_status("roof", 0.29, CONFIG), "not_built")

    def test_cavity_uses_host_when_occluded(self):
        self.assertEqual(predict_status("cavity", 0.0, CONFIG, yellow=None, host_coverage=0.66), "built")
        self.assertEqual(predict_status("cavity", 0.0, CONFIG, yellow=None, host_coverage=0.0), "not_built")

    def test_cavity_yellow_overrides_absent_host(self):
        self.assertEqual(predict_status("cavity", 0.4, CONFIG, yellow=0.8, host_coverage=0.0), "built")
        self.assertEqual(predict_status("cavity", 0.785, CONFIG, yellow=0.1, host_coverage=0.0), "not_built")

    def test_unobserved_coverage_stays_unobserved(self):
        self.assertEqual(predict_status("exterior", None, CONFIG), "unobserved")


if __name__ == "__main__":
    unittest.main()