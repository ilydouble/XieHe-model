import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("compare_pose_models", SCRIPTS / "compare_pose_models.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ComparePoseModelsTests(unittest.TestCase):
    def prediction(self, confidence=0.9, points=None):
        return MODULE.PosePrediction(
            1000,
            2000,
            (100.0, 100.0, 900.0, 1900.0),
            confidence,
            points or ((200.0, 300.0), (800.0, 300.0), (200.0, 1100.0), (800.0, 1100.0), (250.0, 1700.0), (750.0, 1700.0)),
            (0.9,) * 6,
        )

    def test_online_accepts_well_spread_prediction(self):
        accepted, reason = MODULE.online_accept_prediction(self.prediction())
        self.assertIsNone(reason)
        self.assertEqual(accepted.box_confidence, 0.9)

    def test_online_rejects_low_box_and_collapsed_y(self):
        accepted, reason = MODULE.online_accept_prediction(self.prediction(confidence=0.49))
        self.assertEqual(reason, "box_confidence")
        self.assertFalse(accepted.has_keypoints)
        collapsed = tuple((100.0 + index * 120.0, 500.0 + index) for index in range(6))
        accepted, reason = MODULE.online_accept_prediction(self.prediction(points=collapsed))
        self.assertEqual(reason, "collapsed_y")
        self.assertFalse(accepted.has_keypoints)

    def test_legacy_swap_exchanges_each_lr_pair(self):
        prediction = self.prediction()
        swapped = MODULE.apply_lr_swap(prediction)
        self.assertEqual(swapped.keypoints_xy[0], prediction.keypoints_xy[1])
        self.assertEqual(swapped.keypoints_xy[2], prediction.keypoints_xy[3])
        self.assertEqual(swapped.keypoints_xy[4], prediction.keypoints_xy[5])

    def test_truth_span_uses_shoulder_and_lower_group_means(self):
        label = MODULE.PoseLabel(((0.2, 0.1, 2), (0.8, 0.1, 2), (0.2, 0.5, 2), (0.8, 0.5, 2), (0.2, 0.7, 2), (0.8, 0.7, 2)))
        self.assertAlmostEqual(MODULE.truth_span_px(label, 1000), 500.0)


if __name__ == "__main__":
    unittest.main()
