#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "6-train_ap_model" / "train_pose.py"
SPEC = importlib.util.spec_from_file_location("train_pose", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainPoseRoiConfigTest(unittest.TestCase):
    def test_defaults_preserve_existing_training(self):
        args = MODULE.parse_args([])
        self.assertEqual(args.data, "pose_data.yaml")
        self.assertEqual(args.augmentation_profile, "standard")
        config = MODULE.augmentation_config("standard")
        self.assertTrue(config["multi_scale"])
        self.assertEqual(config["mosaic"], 1.0)
        self.assertEqual(config["scale"], 0.5)

    def test_roi_low_disables_confounding_augmentations(self):
        config = MODULE.augmentation_config("roi_low")
        self.assertFalse(config["multi_scale"])
        self.assertEqual(config["mosaic"], 0.0)
        self.assertEqual(config["mixup"], 0.0)
        self.assertEqual(config["copy_paste"], 0.0)
        self.assertEqual(config["erasing"], 0.0)
        self.assertEqual(config["scale"], 0.15)

    def test_resolve_data_relative_to_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            script_dir = Path(temporary)
            expected = script_dir / "dataset.yaml"
            expected.write_text("nc: 1\n", encoding="utf-8")
            self.assertEqual(MODULE.resolve_data_yaml("dataset.yaml", script_dir), expected.resolve())

    def test_mixed_yaml_uses_two_train_roots_and_raw_eval(self):
        config_path = SCRIPT.parent / "pose_data_roi_mixed.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["train"], ["pose_data/images/train", "pose_roi_views/images/train"])
        self.assertEqual(config["val"], "pose_data/images/val")
        self.assertEqual(config["test"], "pose_data/images/test")
        self.assertEqual(config["kpt_shape"], [6, 3])


if __name__ == "__main__":
    unittest.main()
