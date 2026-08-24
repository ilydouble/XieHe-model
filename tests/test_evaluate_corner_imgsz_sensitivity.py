#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_corner_imgsz_sensitivity", SCRIPTS / "evaluate_corner_imgsz_sensitivity.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def obj(class_id: int, area: float = 0.01):
    side = area**0.5
    return MODULE.corner_eval.CornerObject(
        class_id,
        (0.0, 0.0, side * 100, side * 100),
        ((1.0, 1.0, 2.0), (2.0, 1.0, 2.0), (2.0, 2.0, 2.0), (1.0, 2.0, 2.0)),
    )


def metric_row(class_id: int, distance: float | None):
    return {
        "class_id": class_id,
        "visible": True,
        "distance_px": distance,
    }


class CornerImgszSensitivityTests(unittest.TestCase):
    def test_filter_base_truth_ignores_l6_and_t13(self):
        truth = {class_id: obj(class_id) for class_id in (0, 17, 18, 19)}
        self.assertEqual(set(MODULE.filter_base_truth(truth)), {0, 17})

    def test_size_bins_include_missing_points_in_recall_denominator(self):
        samples = [
            {
                "truth_area_fraction": {"0": 0.001, "1": 0.01, "2": 0.1},
                "size_800": {
                    "point_rows": [
                        metric_row(0, 5.0),
                        metric_row(0, None),
                        metric_row(1, 10.0),
                        metric_row(2, 20.0),
                    ]
                },
            }
        ]
        result = MODULE.aggregate_size_bins(samples, "size_800", 0.002, 0.02)
        self.assertEqual(result["small"]["visible_points"], 2)
        self.assertEqual(result["small"]["detected_points"], 1)
        self.assertEqual(result["small"]["point_recall"], 0.5)
        self.assertEqual(result["medium"]["mean_error_px"], 10.0)
        self.assertEqual(result["large"]["mean_error_px"], 20.0)

    def test_select_representatives_covers_worse_middle_and_better(self):
        samples = [
            {"filename": f"sample_{index}.png", "improvement_px": float(index)}
            for index in range(-6, 7)
        ]
        selected = MODULE.select_representatives(samples, count_each=2)
        self.assertEqual({item["group"] for item in selected}, {"worse", "median", "better"})
        self.assertEqual(len({item["filename"] for item in selected}), len(selected))


if __name__ == "__main__":
    unittest.main()
