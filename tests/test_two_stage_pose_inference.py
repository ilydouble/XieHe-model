#!/usr/bin/env python3
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "two_stage_pose_inference.py"
SPEC = importlib.util.spec_from_file_location("two_stage_pose_inference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_result(box, keypoints, box_confidence=0.9, keypoint_confidence=0.8):
    boxes = None if box is None else SimpleNamespace(
        xyxy=np.asarray([box], dtype=float), conf=np.asarray([box_confidence], dtype=float)
    )
    keypoint_data = None if keypoints is None else SimpleNamespace(
        xy=np.asarray([keypoints], dtype=float),
        conf=np.asarray([[keypoint_confidence] * len(keypoints)], dtype=float),
    )
    return SimpleNamespace(boxes=boxes, keypoints=keypoint_data)


class FakeModel:
    def __init__(self, results):
        self.results = list(results)
        self.image_shapes = []

    def predict(self, image, **_kwargs):
        self.image_shapes.append(image.shape[:2])
        return [self.results.pop(0)]


class TwoStagePoseInferenceTest(unittest.TestCase):
    def test_expand_roi_clamps_to_image(self):
        self.assertEqual(MODULE.expand_roi((0, 10, 50, 90), 100, 100, margin=0.2), (0, 0, 60, 100))
        self.assertIsNone(MODULE.expand_roi((10, 10, 5, 20), 100, 100))

    def test_two_stage_remaps_box_and_keypoints(self):
        first_points = [(50 + i * 10, 100 + i * 20) for i in range(6)]
        second_points = [(20 + i * 10, 30 + i * 20) for i in range(6)]
        model = FakeModel(
            [
                fake_result((40, 80, 160, 320), first_points),
                fake_result((10, 20, 130, 260), second_points, box_confidence=0.95),
            ]
        )
        image = np.zeros((400, 200, 3), dtype=np.uint8)
        with patch.object(MODULE.time, "perf_counter", side_effect=(0.0, 0.1, 0.3, 0.4, 0.7, 0.8)):
            result = MODULE.two_stage_predict(model, image, roi_margin=0.1)
        self.assertTrue(result.used_second_stage)
        self.assertEqual(result.roi_xyxy, (28, 56, 172, 344))
        self.assertEqual(model.image_shapes, [(400, 200), (288, 144)])
        self.assertEqual(result.final.box_xyxy, (38.0, 76.0, 158.0, 316.0))
        self.assertEqual(result.final.keypoints_xy[0], (48.0, 86.0))
        self.assertAlmostEqual(result.final.normalized_keypoints()[0][0], 0.24)
        self.assertAlmostEqual(result.final.normalized_keypoints()[0][1], 0.215)
        self.assertAlmostEqual(result.first_inference_ms, 200.0)
        self.assertAlmostEqual(result.second_inference_ms, 300.0)
        self.assertAlmostEqual(result.total_inference_ms, 800.0)
        self.assertEqual(MODULE.result_to_dict(result)["timing_ms"]["second_inference"], result.second_inference_ms)

    def test_low_confidence_falls_back_without_second_call(self):
        points = [(50 + i, 100 + i) for i in range(6)]
        model = FakeModel([fake_result((40, 80, 160, 320), points, box_confidence=0.1)])
        result = MODULE.two_stage_predict(model, np.zeros((400, 200, 3), dtype=np.uint8), minimum_first_box_confidence=0.25)
        self.assertFalse(result.used_second_stage)
        self.assertEqual(result.fallback_reason, "first_box_low_confidence")
        self.assertIs(result.final, result.first)
        self.assertEqual(len(model.image_shapes), 1)

    def test_missing_second_detection_returns_first(self):
        points = [(50 + i, 100 + i) for i in range(6)]
        model = FakeModel([fake_result((40, 80, 160, 320), points), fake_result(None, None)])
        result = MODULE.two_stage_predict(model, np.zeros((400, 200, 3), dtype=np.uint8))
        self.assertFalse(result.used_second_stage)
        self.assertEqual(result.fallback_reason, "second_detection_missing")
        self.assertEqual(result.final, result.first)

    def test_dedicated_second_model_is_used_for_roi(self):
        first_points = [(50 + i, 100 + i) for i in range(6)]
        second_points = [(20 + i, 30 + i) for i in range(6)]
        first_model = FakeModel([fake_result((40, 80, 160, 320), first_points)])
        second_model = FakeModel([fake_result((10, 20, 130, 260), second_points)])
        result = MODULE.two_stage_predict(
            first_model,
            np.zeros((400, 200, 3), dtype=np.uint8),
            roi_margin=0.1,
            second_model=second_model,
        )
        self.assertTrue(result.used_second_stage)
        self.assertEqual(first_model.image_shapes, [(400, 200)])
        self.assertEqual(second_model.image_shapes, [(288, 144)])
        self.assertEqual(result.final.keypoints_xy[0], (48.0, 86.0))


if __name__ == "__main__":
    unittest.main()
