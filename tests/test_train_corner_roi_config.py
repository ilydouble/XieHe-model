#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "6-train_ap_model" / "train_corner.py"
SPEC = importlib.util.spec_from_file_location("train_corner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainCornerRoiConfigTest(unittest.TestCase):
    def test_defaults_preserve_existing_training(self):
        args = MODULE.parse_args([])
        self.assertEqual(args.data, "corner_data.yaml")
        self.assertEqual(args.augmentation_profile, "standard")
        config = MODULE.augmentation_config("standard")
        self.assertTrue(config["multi_scale"])
        self.assertEqual(config["mosaic"], 1.0)

    def test_roi_low_disables_confounding_augmentations(self):
        config = MODULE.augmentation_config("roi_low")
        self.assertFalse(config["multi_scale"])
        self.assertEqual(config["mosaic"], 0.0)
        self.assertEqual(config["mixup"], 0.0)
        self.assertEqual(config["copy_paste"], 0.0)
        self.assertEqual(config["erasing"], 0.0)
        self.assertIsNone(config["auto_augment"])

    def test_resolve_data_yaml_supports_absolute_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            yaml_path = Path(temporary) / "data.yaml"
            yaml_path.write_text("path: .\n", encoding="utf-8")
            self.assertEqual(MODULE.resolve_data_yaml(str(yaml_path), SCRIPT.parent), yaml_path.resolve())

    def test_corner_yamls_define_optional_l6_and_t13_classes(self):
        for name in ("corner_data.yaml", "corner_data_roi_mixed.yaml"):
            data = yaml.safe_load((SCRIPT.parent / name).read_text(encoding="utf-8"))
            self.assertEqual(data["nc"], 20)
            self.assertEqual(data["names"], {index: f"V{index}" for index in range(20)})


if __name__ == "__main__":
    unittest.main()
