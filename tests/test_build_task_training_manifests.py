from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_task_training_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_task_training_manifests", SCRIPT)
assert SPEC and SPEC.loader
manifests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifests)


def make_annotation(image_id: int, *, shift: float = 0.0, source: str = "manual") -> dict:
    items = []
    for index, label in enumerate(manifests.SIX_POINT_LABELS):
        pair_index = index // 2
        is_right = label.endswith("R")
        items.append(
            {
                "label": label,
                "type": "point",
                "source": source,
                "point": {
                    "x": (0.6 if is_right else 0.4) + shift,
                    "y": 0.2 + pair_index * 0.3,
                },
            }
        )
    for vertebra_index, vertebra in enumerate(manifests.VERTEBRA_NAMES):
        top = 0.05 + vertebra_index * 0.045
        coords = ((0.4, top), (0.6, top), (0.4, top + 0.02), (0.6, top + 0.02))
        for corner, (x, y) in enumerate(coords, start=1):
            point = {"x": x + shift, "y": y}
            items.append(
                {
                    "label": f"{vertebra}-{corner}",
                    "type": "vertebra",
                    "source": source,
                    "corners": [dict(point) for _ in range(4)],
                }
            )
    return {
        "imageId": image_id,
        "originalFilename": "source.png",
        "imageWidth": 100,
        "imageHeight": 200,
        "vertebrae": items,
    }


def write_annotation(root: Path, image_id: int, data: dict) -> str:
    name = f"{image_id}_case_label.json"
    (root / name).write_text(json.dumps(data), encoding="utf-8")
    return name


class BuildTaskTrainingManifestsTests(unittest.TestCase):
    def test_nonduplicate_manual_is_trainable_for_both_tasks(self) -> None:
        data = make_annotation(1)
        six = manifests.extract_task(data, "six_point")
        spine = manifests.extract_task(data, "spine_pose")
        self.assertFalse(manifests.fatal_reasons(six, "六点"))
        self.assertFalse(manifests.fatal_reasons(spine, "脊柱"))
        self.assertTrue(manifests.source_eligible(six, "six_point"))
        self.assertTrue(manifests.source_eligible(spine, "spine_pose"))

    def test_small_manual_duplicate_selects_one_representative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_annotation(root, 1, make_annotation(1))
            second = write_annotation(root, 2, make_annotation(2, shift=0.001))
            audit = {
                "exact_duplicate_groups": [
                    {
                        "sha256": "abc",
                        "files": ["1_case.png", "2_case.png"],
                    }
                ]
            }
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            output = root / "output"
            summary = manifests.build_manifests(root, audit_path, output)
            self.assertEqual(summary["six_point"]["decisions"], {"可训练": 1, "排除": 1})
            self.assertEqual(summary["spine_pose"]["decisions"], {"可训练": 1, "排除": 1})
            self.assertTrue((output / "六点模型样本清单.csv").exists())
            self.assertIn(first, {first, second})

    def test_large_duplicate_conflict_requires_review(self) -> None:
        annotations = {
            "1_case_label.json": make_annotation(1),
            "2_case_label.json": make_annotation(2, shift=0.03),
        }
        group = {"sha256": "abc", "files": ["1_case.png", "2_case.png"]}
        rows = manifests.classify_task(
            task="six_point",
            annotation_data=annotations,
            duplicate_groups=[group],
            threshold=0.005,
            accepted_six_anomalies=set(),
        )
        self.assertEqual([row["判定"] for row in rows], ["待复核", "待复核"])

    def test_incomplete_task_is_excluded_without_blocking_other_task(self) -> None:
        data = make_annotation(1)
        data["vertebrae"] = [
            item for item in data["vertebrae"] if item["label"] not in manifests.SIX_POINT_LABELS
        ]
        annotations = {"1_case_label.json": data}
        six_rows = manifests.classify_task(
            task="six_point",
            annotation_data=annotations,
            duplicate_groups=[],
            threshold=0.005,
            accepted_six_anomalies=set(),
        )
        spine_rows = manifests.classify_task(
            task="spine_pose",
            annotation_data=annotations,
            duplicate_groups=[],
            threshold=0.005,
            accepted_six_anomalies=set(),
        )
        self.assertEqual(six_rows[0]["判定"], "排除")
        self.assertEqual(spine_rows[0]["判定"], "可训练")

    def test_all_ai_requires_review(self) -> None:
        data = make_annotation(1, source="ai")
        annotations = {"1_case_label.json": data}
        for task in ("six_point", "spine_pose"):
            rows = manifests.classify_task(
                task=task,
                annotation_data=annotations,
                duplicate_groups=[],
                threshold=0.005,
                accepted_six_anomalies=set(),
            )
            self.assertEqual(rows[0]["判定"], "待复核")


if __name__ == "__main__":
    unittest.main()
