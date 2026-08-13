#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_six_point_model_review.py"
SPEC = importlib.util.spec_from_file_location("build_six_point_model_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SixPointModelReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.images = root / "images"
        self.labels = root / "labels"
        self.output = root / "output"
        self.images.mkdir()
        self.labels.mkdir()
        self.model = root / "best.pt"
        self.model.write_bytes(b"fake-model")
        Image.new("L", (200, 100), 90).save(self.images / "sample.png")
        points = " ".join(f"{x} {y} 2" for x, y in ((0.2, 0.2), (0.8, 0.2), (0.25, 0.55), (0.75, 0.55), (0.4, 0.8), (0.6, 0.8)))
        (self.labels / "sample.txt").write_text(f"0 0.5 0.5 0.8 0.8 {points}\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_parse_and_metrics(self):
        label = MODULE.parse_pose_label(self.labels / "sample.txt")
        prediction = MODULE.Prediction(tuple((x + 0.01, y, 0.9) for x, y, _ in label.keypoints), 0.95)
        metrics = MODULE.calculate_sample_metrics(label, prediction, 200, 100)
        self.assertEqual(metrics["detected_visible_count"], 6)
        self.assertAlmostEqual(metrics["mean_error_px"], 2.0)
        self.assertEqual(metrics["pck_20px_hits"], 6)

    def test_build_offline_package(self):
        label = MODULE.parse_pose_label(self.labels / "sample.txt")

        def predictor(_):
            return MODULE.Prediction(tuple((x, y, 0.99) for x, y, _ in label.keypoints), 0.98)

        manifest = MODULE.build_package(self.images, self.labels, self.model, self.output, predictor)
        self.assertEqual(manifest["summary"]["sample_count"], 1)
        self.assertEqual(manifest["summary"]["pck_20px"], 1.0)
        for name in ("打开人工核验页面.html", "review_data.js", "manifest.json", "README.md", "样本索引与自动误差.csv"):
            self.assertTrue((self.output / name).is_file())
        self.assertEqual(len(list((self.output / "previews").glob("*.jpg"))), 1)
        parsed = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(parsed["samples"][0]["metrics"]["mean_error_px"], 0.0)
        html = (self.output / "打开人工核验页面.html").read_text(encoding="utf-8")
        self.assertIn("导出人工结果 CSV", html)
        self.assertIn("localStorage", html)


if __name__ == "__main__":
    unittest.main()
