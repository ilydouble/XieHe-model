#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "6-train_ap_model/train_pose_stage2.py"
SPEC = importlib.util.spec_from_file_location("train_pose_stage2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainPoseStage2Test(unittest.TestCase):
    def test_defaults_are_low_lr_full_model_finetuning(self):
        args = MODULE.parse_args([])
        self.assertEqual(args.data, "pose_data_stage2_existing_roi.yaml")
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.lr0, 0.0003)
        self.assertEqual(args.freeze, 0)
        self.assertEqual(args.save_period, 10)
        with tempfile.TemporaryDirectory() as temporary:
            config = MODULE.build_train_config(args, Path(temporary) / "data.yaml", Path(temporary) / "runs")
        self.assertEqual(config["optimizer"], "AdamW")
        self.assertEqual(config["lr0"], 0.0003)
        self.assertEqual(config["save_period"], 10)
        self.assertNotIn("freeze", config)
        self.assertEqual(config["mosaic"], 0.0)
        self.assertEqual(config["mixup"], 0.0)
        self.assertEqual(config["scale"], 0.10)

    def test_optional_freeze_is_explicit(self):
        args = MODULE.parse_args(["--freeze", "10"])
        config = MODULE.build_train_config(args, Path("data.yaml"), Path("runs"))
        self.assertEqual(config["freeze"], 10)

    def test_default_yaml_reuses_existing_roi_without_test(self):
        config = yaml.safe_load((SCRIPT.parent / "pose_data_stage2_existing_roi.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["path"], "../datasets")
        self.assertEqual(config["train"], "pose_roi_views/images/train")
        self.assertEqual(config["val"], "pose_data/images/val")
        self.assertNotIn("test", config)
        self.assertEqual(config["kpt_shape"], [6, 3])

    def test_predicted_roi_yaml_remains_optional(self):
        config = yaml.safe_load((SCRIPT.parent / "pose_data_stage2_roi.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["path"], "../datasets/pose_stage2_roi")
        self.assertEqual(config["train"], "images/train")
        self.assertEqual(config["val"], "images/val")

    def test_dry_run_does_not_import_ultralytics_or_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "best.pt"
            data = root / "data.yaml"
            weights.write_bytes(b"weights")
            data.write_text("nc: 1\n", encoding="utf-8")
            MODULE.main(["--weights", str(weights), "--data", str(data), "--dry-run"])


if __name__ == "__main__":
    unittest.main()
