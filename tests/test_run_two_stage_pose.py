#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "8-test_model" / "run_two_stage_pose.py"
SPEC = importlib.util.spec_from_file_location("run_two_stage_pose", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_result(box, points, box_confidence=0.9):
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=np.asarray([box], dtype=float), conf=np.asarray([box_confidence], dtype=float)),
        keypoints=SimpleNamespace(xy=np.asarray([points], dtype=float), conf=np.asarray([[0.8] * 6], dtype=float)),
    )


class FakeYOLO:
    def __init__(self, _model_path):
        points1 = [(30, 40), (70, 40), (35, 100), (65, 100), (45, 130), (55, 130)]
        points2 = [(10, 10), (50, 10), (15, 70), (45, 70), (25, 100), (35, 100)]
        self.results = [fake_result((20, 30, 80, 150), points1), fake_result((5, 5, 65, 125), points2)]

    def predict(self, _image, **_kwargs):
        return [self.results.pop(0)]


class RunTwoStagePoseTest(unittest.TestCase):
    def test_cli_accepts_dedicated_second_model(self):
        argv = [
            "run_two_stage_pose.py",
            "--image", "sample.png",
            "--model", "first.pt",
            "--second-model", "second.pt",
            "--output-dir", "out",
        ]
        with patch.object(sys, "argv", argv):
            args = MODULE.parse_args()
        self.assertEqual(args.model, Path("first.pt"))
        self.assertEqual(args.second_model, Path("second.pt"))

    def test_metrics_preserve_signed_vertical_direction(self):
        prediction = MODULE.PosePrediction(
            100,
            200,
            (20, 20, 80, 180),
            0.9,
            tuple((20 + index, 50 + index) for index in range(6)),
            (0.8,) * 6,
        )
        truth = tuple((20 + index, 55 + index, 2) for index in range(6))
        metrics = MODULE.prediction_metrics(prediction, truth)
        self.assertEqual(metrics["detected"], 6)
        self.assertAlmostEqual(metrics["shoulder_mean_dy_px"], -5.0)
        self.assertAlmostEqual(metrics["lower_mean_dy_px"], -5.0)

    def test_aggregate_handles_samples_without_detected_points(self):
        empty_metrics = {
            "detected": 0,
            "mean_error_px": None,
            "mean_dy_px": None,
            "shoulder_mean_dy_px": None,
            "lower_mean_dy_px": None,
        }
        summary = MODULE.aggregate([{"final_metrics": empty_metrics}], "final_metrics")
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["all_six_detected"], 0)
        self.assertIsNone(summary["mean_error_px"])
        self.assertIsNone(summary["shoulder_mean_dy_px"])

    def test_cli_writes_preview_json_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.png"
            output_dir = root / "output"
            cv2.imwrite(str(image_path), np.full((200, 100, 3), 80, dtype=np.uint8))
            fake_ultralytics = types.ModuleType("ultralytics")
            fake_ultralytics.YOLO = FakeYOLO
            argv = [
                "run_two_stage_pose.py",
                "--image", str(image_path),
                "--model", str(root / "model.pt"),
                "--output-dir", str(output_dir),
            ]
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}), patch.object(sys, "argv", argv):
                MODULE.main()
            payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["sample_count"], 1)
            self.assertEqual(payload["summary"]["second_stage_used"], 1)
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertEqual(len(list((output_dir / "previews").glob("*.jpg"))), 1)


if __name__ == "__main__":
    unittest.main()
