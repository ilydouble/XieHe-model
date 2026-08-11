from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_out_of_range_review.py"
SPEC = importlib.util.spec_from_file_location("render_out_of_range_review", SCRIPT)
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class RenderOutOfRangeReviewTests(unittest.TestCase):
    def test_renders_off_canvas_point_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (100, 200), "white").save(root / "1_case.png")
            annotation = {
                "imageId": 1,
                "originalFilename": "case.jpg",
                "imageWidth": 100,
                "imageHeight": 200,
                "vertebrae": [
                    {"label": "CL", "type": "point", "source": "ai", "point": {"x": -0.1, "y": 0.2}},
                    {"label": "CR", "type": "point", "source": "manual", "point": {"x": 0.8, "y": 0.2}},
                ],
            }
            (root / "1_case_label.json").write_text(json.dumps(annotation), encoding="utf-8")
            audit = {
                "issues": [
                    {
                        "code": "coordinate_out_of_range",
                        "file": "1_case_label.json",
                        "message": "CL.point 超出归一化坐标范围 [0,1]",
                    }
                ]
            }
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            output = root / "output"
            summary = review.build_review_package(root, audit_path, output)

            self.assertEqual(summary["rendered_annotations"], 1)
            self.assertEqual(summary["rendered_out_of_range_points"], 1)
            preview = output / "1_case_越界点扩展画布.png"
            self.assertTrue(preview.is_file())
            with Image.open(preview) as rendered:
                self.assertGreater(rendered.width, 100)
            self.assertIn("CL", (output / "越界点索引.csv").read_text(encoding="utf-8-sig"))
            self.assertTrue((output / "打开此文件人工复核.html").is_file())


if __name__ == "__main__":
    unittest.main()
