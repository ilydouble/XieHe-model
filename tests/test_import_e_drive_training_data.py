from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_e_drive_training_data.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("import_e_drive_training_data", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def annotation() -> dict:
    items = []
    six = {
        "CL": (0.2, 0.2), "CR": (0.8, 0.2),
        "IL": (0.25, 0.75), "IR": (0.75, 0.75),
        "SL": (0.45, 0.8), "SR": (0.55, 0.8),
    }
    for label, (x, y) in six.items():
        items.append({"label": label, "type": "point", "source": "manual", "point": {"x": x, "y": y}})
    for vertebra_index, vertebra in enumerate(builder.VERTEBRA_NAMES):
        top = 0.03 + vertebra_index * 0.045
        coords = {1: (0.4, top), 2: (0.6, top), 3: (0.4, top + 0.02), 4: (0.6, top + 0.02)}
        for corner, (x, y) in coords.items():
            point = {"x": x, "y": y}
            items.append({"label": f"{vertebra}-{corner}", "type": "vertebra", "source": "manual", "corners": [dict(point) for _ in range(4)]})
    return {"imageId": 1, "originalFilename": "case.png", "imageWidth": 100, "imageHeight": 200, "vertebrae": items}


def write_manifests(root: Path) -> None:
    root.mkdir(parents=True)
    fields = ["判定", "图像文件", "标注文件", "重复组SHA256", "组内代表样本"]
    row = {"判定": "可训练", "图像文件": "1_case.png", "标注文件": "1_case_label.json", "重复组SHA256": "abc", "组内代表样本": "是"}
    for name in ("六点模型样本清单.csv", "脊柱Pose模型样本清单.csv"):
        with (root / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerow(row)
    (root / "清单汇总.json").write_text(
        json.dumps({"rules": {"accepted_six_anomalies": []}}), encoding="utf-8"
    )


def make_mixed_lr(data: dict) -> dict:
    for item in data["vertebrae"]:
        if item.get("label") in {"CR", "CL"}:
            item["point"]["x"] = 0.1 if item["label"] == "CR" else 0.9
    return data


class ImportTrainingDataTests(unittest.TestCase):
    def test_converts_expected_keypoint_orders(self) -> None:
        data = annotation()
        six = builder.six_point_yolo(data).split()
        self.assertEqual(six[5:11], ["0.80000000", "0.20000000", "2", "0.20000000", "0.20000000", "2"])
        corner = builder.corner_yolo(data).splitlines()[0].split()
        points = corner[5:]
        self.assertEqual(points[0:3], ["0.40000000", "0.03000000", "2"])
        self.assertEqual(points[6:9], ["0.60000000", "0.05000000", "2"])
        self.assertEqual(points[9:12], ["0.40000000", "0.05000000", "2"])

    def test_mirror_policy_transforms_image_and_preserves_keypoint_identity(self) -> None:
        values = builder.six_point_yolo(annotation(), lr_policy="mirror_image").split()
        self.assertEqual(values[5:11], ["0.20000000", "0.20000000", "2", "0.80000000", "0.20000000", "2"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"; destination = root / "destination.png"
            image = Image.new("L", (3, 1)); image.putdata([0, 127, 255]); image.save(source)
            builder.ImageOps.mirror(image).save(destination)
            self.assertTrue(builder.mirrored_pixels_equal(source, destination))

    def test_batch_lr_policy_skips_mixed_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); export = root / "export"; export.mkdir()
            Image.new("L", (100, 200), 128).save(export / "1_case.png")
            data = make_mixed_lr(annotation())
            (export / "1_case_label.json").write_text(json.dumps(data), encoding="utf-8")
            manifests = root / "manifests"; write_manifests(manifests)
            (manifests / "清单汇总.json").write_text(
                json.dumps({"rules": {"accepted_six_anomalies": ["1_case_label.json"]}}),
                encoding="utf-8",
            )
            pose = root / "pose"; corner = root / "corner"
            for target in (pose, corner):
                for split in ("train", "val", "test"):
                    (target / "images" / split).mkdir(parents=True)
                    (target / "labels" / split).mkdir(parents=True)
            result = builder.build(
                export, manifests, pose, corner, root / "output",
                tasks=("six_point",), six_lr_policy="swap_pairs",
            )
            self.assertEqual(result["summary"]["six_point"]["statuses"], {"skipped": 1})

    def test_dry_run_and_apply_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); export = root / "export"; export.mkdir()
            Image.new("L", (100, 200), 128).save(export / "1_case.png")
            (export / "1_case_label.json").write_text(json.dumps(annotation()), encoding="utf-8")
            manifests = root / "manifests"; write_manifests(manifests)
            pose = root / "pose"; corner = root / "corner"
            for target in (pose, corner):
                for split in ("train", "val", "test"):
                    (target / "images" / split).mkdir(parents=True)
                    (target / "labels" / split).mkdir(parents=True)

            dry = builder.build(export, manifests, pose, corner, root / "dry")
            self.assertEqual(dry["summary"]["six_point"]["statuses"], {"planned": 1})
            blocked = builder.build(export, manifests, pose, corner, root / "blocked", apply=True)
            self.assertEqual(blocked["requested_mode"], "apply")
            self.assertEqual(blocked["mode"], "dry_run")
            self.assertTrue(blocked["blocked_reasons"])
            applied = builder.build(
                export, manifests, pose, corner, root / "applied", apply=True,
                six_lr_policy="swap_pairs",
            )
            self.assertEqual(applied["summary"]["spine_pose"]["statuses"], {"imported": 1})
            again = builder.build(
                export, manifests, pose, corner, root / "again", apply=True,
                six_lr_policy="swap_pairs",
            )
            self.assertEqual(again["summary"]["six_point"]["statuses"], {"skipped": 1})
            self.assertEqual(len((pose / "labels" / "train" / "eap_1_case.txt").read_text().split()), 23)
            self.assertEqual(len((corner / "labels" / "train" / "eap_1_case.txt").read_text().splitlines()), 18)
            self.assertTrue((root / "dry" / "sha256_cache.json").exists())
            pose_values = (pose / "labels" / "train" / "eap_1_case.txt").read_text().split()
            self.assertEqual(pose_values[5:11], ["0.20000000", "0.20000000", "2", "0.80000000", "0.20000000", "2"])


if __name__ == "__main__":
    unittest.main()
