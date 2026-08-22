#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_pose_black_border_review.py"
SPEC = importlib.util.spec_from_file_location("build_pose_black_border_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PoseBlackBorderReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.dataset = root / "pose_data"
        self.output = root / "output"
        for split in ("train", "val", "test"):
            (self.dataset / "images" / split).mkdir(parents=True)
            (self.dataset / "labels" / split).mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def add_sample(self, split: str, name: str, image: Image.Image, points=None):
        image.save(self.dataset / "images" / split / f"{name}.png")
        points = points or ((0.3, 0.2), (0.7, 0.2), (0.3, 0.55), (0.7, 0.55), (0.4, 0.8), (0.6, 0.8))
        encoded = " ".join(f"{x} {y} 2" for x, y in points)
        (self.dataset / "labels" / split / f"{name}.txt").write_text(
            f"0 0.5 0.5 0.8 0.8 {encoded}\n", encoding="utf-8"
        )

    def test_continuous_border_and_keypoint_risk(self):
        image = Image.new("L", (100, 200), 100)
        ImageDraw.Draw(image).rectangle((0, 0, 19, 199), fill=0)
        metrics = MODULE.measure_image(image)
        self.assertEqual(metrics.left, 20)
        self.assertAlmostEqual(metrics.border_area_fraction, 0.2)
        self.assertEqual(MODULE.selection_reason(metrics), "continuous_ge_5pct")
        risky = MODULE.keypoints_in_border(((0.1, 0.5, 2), (0.5, 0.5, 2)), metrics)
        self.assertEqual(risky, ["CR"])

    def test_wide_interrupted_canvas_rule(self):
        image = Image.new("L", (240, 100), 0)
        draw = ImageDraw.Draw(image)
        draw.rectangle((90, 0, 149, 99), fill=120)
        draw.rectangle((0, 20, 239, 24), fill=255)
        metrics = MODULE.measure_image(image)
        self.assertLess(metrics.border_area_fraction, 0.05)
        self.assertEqual(MODULE.selection_reason(metrics), "wide_interrupted_canvas")

    def test_build_offline_package(self):
        bordered = Image.new("L", (100, 200), 100)
        ImageDraw.Draw(bordered).rectangle((0, 0, 14, 199), fill=0)
        self.add_sample("train", "eap_bordered", bordered, points=((0.1, 0.2), (0.7, 0.2), (0.3, 0.55), (0.7, 0.55), (0.4, 0.8), (0.6, 0.8)))
        self.add_sample("val", "plain", Image.new("L", (100, 200), 100))
        manifest = MODULE.build_package(self.dataset, self.output)
        self.assertEqual(manifest["summary"]["dataset_image_count"], 2)
        self.assertEqual(manifest["summary"]["candidate_count"], 1)
        self.assertEqual(manifest["summary"]["keypoint_crop_risk_count"], 1)
        for name in ("打开黑边核验页面.html", "review_data.js", "manifest.json", "README.md", "样本索引.csv"):
            self.assertTrue((self.output / name).is_file())
        self.assertEqual(len(list((self.output / "previews").glob("*.jpg"))), 1)
        parsed = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(parsed["samples"][0]["reason"], "continuous_ge_5pct")
        self.assertEqual(parsed["samples"][0]["risky_keypoints"], ["CR"])


if __name__ == "__main__":
    unittest.main()
