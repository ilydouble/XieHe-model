#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_corner_models.py"
SPEC = importlib.util.spec_from_file_location("compare_corner_models", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def obj(class_id, y, conf=0.9, x=10.0):
    return MODULE.CornerObject(class_id, (x, y, x + 10, y + 10), ((x, y, 2), (x + 10, y, 2), (x + 10, y + 10, 2), (x, y + 10, 2)), conf)


class CompareCornerModelsTest(unittest.TestCase):
    def test_parse_corner_label_scales_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "label.txt"
            path.write_text("0 0.5 0.5 0.2 0.4 0.4 0.3 2 0.6 0.3 2 0.6 0.7 2 0.4 0.7 2\n", encoding="utf-8")
            parsed = MODULE.parse_corner_label(path, 100, 200)[0]
        self.assertEqual(parsed.box_xyxy, (40.0, 60.0, 60.0, 140.0))
        self.assertEqual(parsed.keypoints[0], (40.0, 60.0, 2.0))

    def test_production_assignments_nms_then_y_sort(self):
        upper = obj(8, 10, 0.8)
        upper_duplicate = obj(3, 11, 0.7)
        lower = obj(1, 50, 0.9)
        assigned = MODULE.production_assignments([lower, upper_duplicate, upper], 0.5, 0.3)
        self.assertEqual(list(assigned), [0, 1])
        self.assertEqual(assigned[0].keypoints, upper.keypoints)
        self.assertEqual(assigned[1].keypoints, lower.keypoints)

    def test_native_assignments_uses_best_per_class(self):
        assigned = MODULE.native_assignments([obj(2, 20, 0.6), obj(2, 22, 0.9), obj(3, 30, 0.4)], 0.5)
        self.assertEqual(set(assigned), {2})
        self.assertEqual(assigned[2].confidence, 0.9)

    def test_evaluate_assignments_counts_missing_as_recall_failure(self):
        truth = {0: obj(0, 10), 1: obj(1, 30)}
        assigned = {0: obj(0, 10)}
        metrics = MODULE.evaluate_assignments(truth, assigned, 100, 100)
        self.assertEqual(metrics["visible_points"], 8)
        self.assertEqual(metrics["detected_points"], 4)
        self.assertEqual(metrics["missing_vertebrae"], [1])
        self.assertEqual(metrics["mean_error_px"], 0.0)

    def test_package_hashes_exclude_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text("changing", encoding="utf-8")
            (root / "report.txt").write_text("stable", encoding="utf-8")
            hashes = MODULE.package_file_hashes(root)
        self.assertEqual(set(hashes), {"report.txt"})
        self.assertEqual(len(hashes["report.txt"]), 64)

    def test_source_group(self):
        self.assertEqual(MODULE.source_group("eap_1.png"), "eap")
        self.assertEqual(MODULE.source_group("1.2.156.png"), "server_uid")
        self.assertEqual(MODULE.source_group("123__CR.png"), "legacy_numeric")
        self.assertEqual(MODULE.source_group("WZSY_sample.png"), "new_site_code")


if __name__ == "__main__":
    unittest.main()
