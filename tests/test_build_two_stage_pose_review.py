#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_two_stage_pose_review.py"
SPEC = importlib.util.spec_from_file_location("build_two_stage_pose_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TwoStagePoseReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.images = root / "images"
        self.labels = root / "labels"
        self.output = root / "output"
        self.images.mkdir()
        self.labels.mkdir()
        self.model = root / "run" / "weights" / "best.pt"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"fake-model")
        (self.model.parents[1] / "args.yaml").write_text("imgsz: 800\n", encoding="utf-8")
        self.second_model = root / "stage2" / "weights" / "best.pt"
        self.second_model.parent.mkdir(parents=True)
        self.second_model.write_bytes(b"fake-second-model")
        (self.second_model.parents[1] / "args.yaml").write_text("imgsz: 800\nlr0: 0.001\n", encoding="utf-8")
        Image.new("L", (200, 400), 80).save(self.images / "eap_sample.png")
        self.points = ((0.3, 0.2), (0.7, 0.2), (0.32, 0.65), (0.68, 0.65), (0.45, 0.75), (0.55, 0.75))
        encoded = " ".join(f"{x} {y} 2" for x, y in self.points)
        (self.labels / "eap_sample.txt").write_text(f"0 0.5 0.475 0.4 0.55 {encoded}\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def prediction(self, dy: float):
        return MODULE.PosePrediction(
            200,
            400,
            (30, 50, 170, 330),
            0.95,
            tuple((x * 200, y * 400 + dy) for x, y in self.points),
            (0.99,) * 6,
        )

    def result(self):
        return MODULE.TwoStageResult(
            self.prediction(-5),
            self.prediction(-2),
            True,
            None,
            (20, 30, 180, 360),
            10.0,
            8.0,
            18.5,
        )

    def test_signed_metrics_and_span(self):
        label = MODULE.parse_pose_label(self.labels / "eap_sample.txt")
        metrics = MODULE.calculate_stage_metrics(label, self.prediction(-5), 200, 400)
        self.assertEqual(metrics["detected_visible_count"], 6)
        self.assertAlmostEqual(metrics["mean_error_px"], 5.0)
        self.assertAlmostEqual(metrics["shoulder_mean_dy_px"], -5.0)
        self.assertAlmostEqual(metrics["lower_mean_dy_px"], -5.0)
        self.assertAlmostEqual(metrics["span_bias_px"], 0.0)

    def test_builds_review_package_with_timing_and_filters(self):
        configuration = {"conf": 0.25, "imgsz": 800, "roi_margin": 0.2, "roi_conf": 0.25, "device": "cpu", "warmup": 1}
        calls = []

        def predictor(_path):
            calls.append(1)
            return self.result()

        manifest = MODULE.build_package(self.images, self.labels, self.model, self.output, predictor, configuration)
        self.assertEqual(len(calls), 2)
        self.assertEqual(manifest["summary"]["sample_count"], 1)
        self.assertEqual(manifest["summary"]["timing"]["first_inference"]["mean_ms"], 10.0)
        self.assertEqual(manifest["summary"]["comparison"]["improved_sample_count"], 1)
        for name in ("README.md", "分析报告.md", "manifest.json", "review_data.js", "打开两阶段评测页面.html", "逐图指标.csv"):
            self.assertTrue((self.output / name).is_file())
        self.assertEqual(len(list((self.output / "previews").glob("*.jpg"))), 1)
        html = (self.output / "打开两阶段评测页面.html").read_text(encoding="utf-8")
        self.assertIn("improvement_asc", html)
        self.assertIn("localStorage", html)
        parsed = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn(parsed["samples"][0]["preview"], parsed["package_files"])

    def test_manifest_tracks_dedicated_second_model(self):
        configuration = {"conf": 0.25, "imgsz": 800, "roi_margin": 0.2, "roi_conf": 0.25, "device": "cpu", "warmup": 0}
        manifest = MODULE.build_package(
            self.images,
            self.labels,
            self.model,
            self.output,
            lambda _path: self.result(),
            configuration,
            second_model_path=self.second_model,
        )
        self.assertEqual(manifest["first_stage_model_sha256"], MODULE.sha256_file(self.model))
        self.assertEqual(manifest["second_stage_model_sha256"], MODULE.sha256_file(self.second_model))
        self.assertNotEqual(manifest["first_stage_model_sha256"], manifest["second_stage_model_sha256"])


if __name__ == "__main__":
    unittest.main()
