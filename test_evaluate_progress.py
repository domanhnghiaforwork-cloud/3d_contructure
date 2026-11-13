import csv
import tempfile
import unittest
from pathlib import Path

from evaluate_progress import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.labels = self.directory / "labels.csv"
        self.predictions = self.directory / "predictions.csv"
        self.write_labels([("a", "built"), ("b", "partial"), ("c", "not_built")])

    def write_labels(self, rows):
        with self.labels.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "label", "split", "label_source"])
            writer.writerows((sample_id, label, "test", "unit-test-fixture") for sample_id, label in rows)

    def write_predictions(self, rows):
        with self.predictions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "prediction"])
            writer.writerows(rows)

    def test_perfect_three_class_fixture(self):
        self.write_predictions([("c", "not_built"), ("a", "built"), ("b", "partial")])
        result = evaluate(self.labels, self.predictions, "test")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["class_support"], {"built": 1, "partial": 1, "not_built": 1})
        self.assertTrue(result["meets_numeric_threshold"])
        self.assertEqual(len(result["reference_sha256"]), 64)

    def test_missing_prediction_cannot_shrink_denominator(self):
        self.write_predictions([("a", "built"), ("b", "partial")])
        with self.assertRaisesRegex(ValueError, "1 missing"):
            evaluate(self.labels, self.predictions, "test")

    def test_extra_prediction_rejected(self):
        self.write_predictions([("a", "built"), ("b", "partial"), ("c", "not_built"), ("d", "built")])
        with self.assertRaisesRegex(ValueError, "1 extra"):
            evaluate(self.labels, self.predictions, "test")

    def test_unobserved_counts_as_incorrect(self):
        self.write_predictions([("a", "built"), ("b", "partial"), ("c", "unobserved")])
        result = evaluate(self.labels, self.predictions, "test")
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["n_samples"], 3)
        self.assertEqual(result["n_unobserved"], 1)
        self.assertEqual(result["confusion_matrix"][2][3], 1)
        self.assertFalse(result["meets_numeric_threshold"])

    def test_one_class_cannot_qualify_for_three_class_goal(self):
        self.write_labels([("a", "built")])
        self.write_predictions([("a", "built")])
        result = evaluate(self.labels, self.predictions, "test")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertFalse(result["all_three_reference_classes_present"])
        self.assertFalse(result["meets_numeric_threshold"])

    def test_high_accuracy_majority_prediction_is_exposed(self):
        rows = [(str(index), "built") for index in range(8)] + [("8", "partial"), ("9", "not_built")]
        self.write_labels(rows)
        self.write_predictions([(sample_id, "built") for sample_id, _ in rows])
        result = evaluate(self.labels, self.predictions, "test")
        self.assertEqual(result["accuracy"], 0.8)
        self.assertEqual(result["majority_baseline_accuracy"], 0.8)
        self.assertFalse(result["beats_majority_baseline"])
        self.assertLess(result["macro_f1"], 0.3)

    def test_duplicate_reference_rejected(self):
        self.write_labels([("a", "built"), ("a", "partial")])
        self.write_predictions([("a", "built")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(self.labels, self.predictions, "test")

    def test_invalid_reference_rejected(self):
        self.write_labels([("a", "unobserved")])
        self.write_predictions([("a", "built")])
        with self.assertRaisesRegex(ValueError, "Invalid reference"):
            evaluate(self.labels, self.predictions, "test")

    def test_empty_split_rejected(self):
        self.write_predictions([("a", "built")])
        with self.assertRaisesRegex(ValueError, "No reference labels"):
            evaluate(self.labels, self.predictions, "validation")


if __name__ == "__main__":
    unittest.main()
