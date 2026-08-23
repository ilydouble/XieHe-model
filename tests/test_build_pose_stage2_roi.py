#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "build_pose_stage2_roi.py"
SPEC = importlib.util.spec_from_file_location("build_pose_stage2_roi", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildPoseStage2RoiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.dataset = root / "pose_data"
        self.output = root / "pose_stage2_roi"
        self.model = root / "best.pt"
        self.model.write_bytes(b"fake-stage-one-model")
        for split in ("train", "val"):
            (self.dataset / "images" / split).mkdir(parents=True)
            (self.dataset / "labels" / split).mkdir(parents=True)
            Image.new("L", (200, 400), 80).save(self.dataset / "images" / split / f"{split}_sample.png")
            points = ((0.30, 0.20), (0.70, 0.20), (0.32, 0.65), (0.68, 0.65), (0.45, 0.75), (0.55, 0.75))
            encoded = " ".join(f"{x} {y} 2" for x, y in points)
            (self.dataset / "labels" / split / f"{split}_sample.txt").write_text(
                f"0 0.5 0.475 0.4 0.55 {encoded}\n", encoding="utf-8"
            )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def predictor(image):
        return MODULE.PosePrediction(
            image.shape[1],
            image.shape[0],
            (50.0, 60.0, 150.0, 330.0),
            0.9,
            ((60.0, 80.0), (140.0, 80.0), (64.0, 260.0), (136.0, 260.0), (90.0, 300.0), (110.0, 300.0)),
            (0.9,) * 6,
        )

    def test_predicted_crop_is_deterministic_and_label_round_trips(self):
        label = MODULE.parse_pose_label(self.dataset / "labels/train/train_sample.txt")
        first = MODULE.predicted_crop_box((50, 60, 150, 330), label, 200, 400, "sample.png", "train", 1)
        second = MODULE.predicted_crop_box((50, 60, 150, 330), label, 200, 400, "sample.png", "train", 1)
        self.assertEqual(first, second)
        box, _ = first
        transformed = MODULE.transform_label(label, box, 200, 400)
        for original, changed in zip(label.keypoints, transformed.keypoints):
            self.assertAlmostEqual(original[0] * 200, changed[0] * box.width + box.left, places=5)
            self.assertAlmostEqual(original[1] * 400, changed[1] * box.height + box.top, places=5)

    def test_plan_and_apply_build_roi_only_train_and_val(self):
        train, skipped_train = MODULE.plan_split(self.dataset, "train", self.predictor, 2, 1, 0.2, 0.06, 0.12, 0.25)
        val, skipped_val = MODULE.plan_split(self.dataset, "val", self.predictor, 1, 1, 0.2, 0.0, 0.0, 0.25)
        records = train + val
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 1)
        manifest = MODULE.apply_dataset(records, skipped_train + skipped_val, self.dataset, self.output, self.model, {"train_variants": 2})
        self.assertEqual(manifest["summary"]["train_roi_count"], 2)
        self.assertEqual(manifest["summary"]["val_roi_count"], 1)
        self.assertFalse((self.output / "images/test").exists())
        self.assertEqual(len(list((self.output / "images/train").glob("*.png"))), 2)
        self.assertEqual(len(list((self.output / "images/val").glob("*.png"))), 1)
        loaded = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(loaded["stage1_model_sha256"], MODULE.sha256_file(self.model))
        for record in loaded["records"]:
            MODULE.validate_transformed_label(MODULE.parse_pose_label(self.output / record["output_label"]))

    def test_missing_prediction_is_skipped(self):
        def missing(image):
            return MODULE.PosePrediction(image.shape[1], image.shape[0], None, 0.0, ((0.0, 0.0),) * 6, (0.0,) * 6)

        records, skipped = MODULE.plan_split(self.dataset, "train", missing, 2, 1, 0.2, 0.06, 0.12, 0.25)
        self.assertEqual(records, [])
        self.assertEqual(skipped[0]["reason"], "prediction_missing")


if __name__ == "__main__":
    unittest.main()
