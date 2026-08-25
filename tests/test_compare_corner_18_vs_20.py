#!/usr/bin/env python3
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "compare_corner_18_vs_20.py"
SPEC = importlib.util.spec_from_file_location("compare_corner_18_vs_20", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def obj(class_id, y=10.0):
    return MODULE.corner_eval.CornerObject(
        class_id,
        (10, y, 20, y + 10),
        ((10, y, 2), (20, y, 2), (20, y + 10, 2), (10, y + 10, 2)),
        0.9,
    )


class CompareCorner18Vs20Test(unittest.TestCase):
    def test_filter_classes_separates_base_and_extras(self):
        values = {0: obj(0), 17: obj(17), 18: obj(18), 19: obj(19)}
        self.assertEqual(set(MODULE.filter_classes(values, MODULE.BASE_CLASS_IDS)), {0, 17})
        self.assertEqual(set(MODULE.filter_classes(values, MODULE.EXTRA_CLASS_IDS)), {18, 19})

    def test_aggregate_rare_metrics_uses_only_selected_truth(self):
        truth = {18: obj(18), 19: obj(19)}
        assigned = {18: obj(18)}
        metric = MODULE.corner_eval.evaluate_assignments(truth, assigned, 100, 100)
        summary = MODULE.aggregate([{"new_extra": metric}], "new_extra")
        self.assertEqual(summary["visible_points"], 8)
        self.assertEqual(summary["detected_points"], 4)
        self.assertEqual(summary["point_recall"], 0.5)

    def test_bootstrap_constant_has_exact_interval(self):
        self.assertEqual(MODULE.bootstrap_mean_ci([2.5] * 10, iterations=50), [2.5, 2.5])

    def test_missing_rare_class_formats_as_not_available(self):
        self.assertEqual(MODULE.format_percent(None), "N/A")
        self.assertEqual(MODULE.format_percent(0.5), "50.00%")

    def test_representatives_always_include_extra_cases(self):
        samples = [
            {"filename": f"{index}.jpg", "base_improvement_px": float(index), "extra_truth_classes": [18] if index == 3 else []}
            for index in range(8)
        ]
        selected = MODULE.select_representatives(samples, count_each=1)
        extra = [item for item in selected if item["filename"] == "3.jpg"]
        self.assertEqual(len(extra), 1)

    def test_interpretation_requires_recall_and_pck_consistency(self):
        old = {"mean_error_px": 10.0, "point_recall": 0.98, "pck_20_all": 0.9}
        new = {"mean_error_px": 9.0, "point_recall": 0.981, "pck_20_all": 0.91}
        self.assertIn("实质提高", MODULE.automatic_interpretation(old, new))

    def test_extra_summary_counts_false_positive_classes(self):
        metric = MODULE.corner_eval.evaluate_assignments({18: obj(18)}, {18: obj(18), 19: obj(19)}, 100, 100)
        samples = [{
            "extra_truth_classes": [18],
            "new_extra_predicted_classes": [18, 19],
            "new_extra": metric,
        }]
        summary = MODULE.summarize_extra_predictions(samples)
        self.assertEqual(summary["detected_vertebrae"], 1)
        self.assertEqual(summary["false_positive_vertebrae"], 1)
        self.assertEqual(summary["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
