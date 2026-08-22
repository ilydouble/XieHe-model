#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_pose_roi_views.py"
SPEC = importlib.util.spec_from_file_location("build_pose_roi_views", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildPoseRoiViewsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.dataset = root / "pose_data"
        self.output = root / "pose_roi_views"
        (self.dataset / "images" / "train").mkdir(parents=True)
        (self.dataset / "labels" / "train").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def add_sample(self, name: str, width: int = 200, height: int = 400):
        Image.new("L", (width, height), 80).save(self.dataset / "images" / "train" / f"{name}.png")
        points = ((0.30, 0.20), (0.70, 0.20), (0.32, 0.65), (0.68, 0.65), (0.45, 0.75), (0.55, 0.75))
        encoded = " ".join(f"{x} {y} 2" for x, y in points)
        (self.dataset / "labels" / "train" / f"{name}.txt").write_text(
            f"0 0.5 0.475 0.4 0.55 {encoded}\n", encoding="utf-8"
        )

    def test_crop_is_deterministic_and_contains_target(self):
        self.add_sample("sample")
        label = MODULE.parse_pose_label(self.dataset / "labels" / "train" / "sample.txt")
        first = MODULE.compute_crop_box(label, 200, 400, "sample.png")
        second = MODULE.compute_crop_box(label, 200, 400, "sample.png")
        self.assertEqual(first, second)
        self.assertLess(first.width * first.height, 200 * 400)
        MODULE.validate_crop_contains_target(first, label, 200, 400)

    def test_transform_round_trip_geometry(self):
        self.add_sample("sample")
        label = MODULE.parse_pose_label(self.dataset / "labels" / "train" / "sample.txt")
        box = MODULE.compute_crop_box(label, 200, 400, "sample.png")
        transformed = MODULE.transform_label(label, box, 200, 400)
        for original, changed in zip(label.keypoints, transformed.keypoints):
            original_px = original[0] * 200, original[1] * 400
            restored_px = changed[0] * box.width + box.left, changed[1] * box.height + box.top
            self.assertAlmostEqual(original_px[0], restored_px[0])
            self.assertAlmostEqual(original_px[1], restored_px[1])

    def test_apply_builds_roi_only_dataset(self):
        self.add_sample("a")
        self.add_sample("b", 240, 480)
        records = MODULE.plan_views(self.dataset)
        manifest = MODULE.apply_plan(records, self.dataset, self.output, 20260822, 0.20, 0.05, 0.10)
        self.assertEqual(manifest["summary"]["source_train_count"], 2)
        self.assertEqual(manifest["summary"]["mixed_train_count"], 4)
        self.assertEqual(len(list((self.output / "images" / "train").glob("roi_*.png"))), 2)
        self.assertEqual(len(list((self.output / "labels" / "train").glob("roi_*.txt"))), 2)
        self.assertFalse((self.output / "images" / "val").exists())
        parsed = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        for record in parsed["records"]:
            self.assertTrue((self.output / record["output_image"]).is_file())
            MODULE.validate_transformed_label(MODULE.parse_pose_label(self.output / record["output_label"]))

    def test_refuses_existing_output(self):
        self.add_sample("sample")
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            MODULE.apply_plan(MODULE.plan_views(self.dataset), self.dataset, self.output, 20260822, 0.20, 0.05, 0.10)


if __name__ == "__main__":
    unittest.main()
