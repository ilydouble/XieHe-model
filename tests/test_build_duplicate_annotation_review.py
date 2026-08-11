import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_duplicate_annotation_review.py"
SPEC = importlib.util.spec_from_file_location("duplicate_review", SCRIPT)
duplicate_review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(duplicate_review)


def write_candidate(root: Path, stem: str, color: int, offset: float) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    image_name = f"{stem}.png"
    annotation_name = f"{stem}_label.json"
    Image.new("L", (40, 80), color).save(root / image_name)
    items = []
    for label, x, y in (("CL", 0.2 + offset, 0.2), ("CR", 0.8, 0.2)):
        items.append({"label": label, "type": "point", "source": "manual", "point": {"x": x, "y": y}})
    for corner in range(1, 5):
        items.append(
            {
                "label": f"T1-{corner}",
                "type": "vertebra",
                "source": "ai",
                "corners": [{"x": 0.4 + offset + corner * 0.01, "y": 0.3 + corner * 0.01}],
            }
        )
    data = {"imageId": stem, "originalFilename": image_name, "vertebrae": items}
    (root / annotation_name).write_text(json.dumps(data), encoding="utf-8")
    return image_name, annotation_name


class DuplicateAnnotationReviewTest(unittest.TestCase):
    def test_task_delta_treats_missing_labels_as_structure_conflict(self):
        candidates = [
            {"points": {"CL": {"x": 0.2, "y": 0.3, "source": "manual"}}},
            {"points": {}},
        ]

        result = duplicate_review.task_delta(candidates, duplicate_review.SIX_POINT_LABELS)

        self.assertEqual(result["max_delta"], 0.0)
        self.assertTrue(result["structure_conflict"])
        self.assertTrue(result["has_conflict"])

    def test_builds_two_and_three_candidate_groups_with_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            current = base / "current"
            fallback = base / "fallback"
            output = base / "output"
            group1 = []
            for index, offset in enumerate((0.0, 0.02), 1):
                image, _ = write_candidate(current, f"a{index}", 80, offset)
                group1.append(image)
            group2 = []
            for index, offset in enumerate((0.0, 0.03, 0.06), 1):
                root = fallback if index == 3 else current
                image, _ = write_candidate(root, f"b{index}", 120, offset)
                group2.append(image)

            hashes = []
            for files, root in ((group1, current), (group2, current)):
                first = root / files[0]
                content = first.read_bytes()
                for name in files[1:]:
                    target = current / name if (current / name).is_file() else fallback / name
                    target.write_bytes(content)
                hashes.append(duplicate_review.sha256_file(first))
            audit = {
                "exact_duplicate_groups": [
                    {"sha256": hashes[0], "files": group1},
                    {"sha256": hashes[1], "files": group2},
                ]
            }
            audit_path = base / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            manifest = duplicate_review.build_package(
                current,
                audit_path,
                output,
                fallback_dirs=[fallback],
                max_image_height=60,
            )

            self.assertEqual(manifest["group_count"], 2)
            self.assertEqual(manifest["candidate_count"], 5)
            self.assertEqual(manifest["group_size_counts"], {"2": 1, "3": 1})
            self.assertTrue((output / "打开核对页面.html").is_file())
            self.assertTrue((output / "review_data.js").is_file())
            self.assertEqual(len(list((output / "images").glob("*.jpg"))), 2)
            html = (output / "打开核对页面.html").read_text(encoding="utf-8")
            self.assertIn("两个/全部都不对", html)
            self.assertIn("导出JSON", html)
            self.assertIn("jumpToNumber", html)


if __name__ == "__main__":
    unittest.main()
