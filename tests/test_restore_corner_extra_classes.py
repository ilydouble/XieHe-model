#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

ROI_SPEC = importlib.util.spec_from_file_location("build_corner_roi_views", SCRIPTS / "build_corner_roi_views.py")
ROI = importlib.util.module_from_spec(ROI_SPEC)
sys.modules[ROI_SPEC.name] = ROI
ROI_SPEC.loader.exec_module(ROI)

SPEC = importlib.util.spec_from_file_location("restore_corner_extra_classes", SCRIPTS / "restore_corner_extra_classes.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(class_id: int, y: float) -> str:
    points = ((0.42, y), (0.58, y), (0.58, y + 0.025), (0.42, y + 0.025))
    payload = " ".join(f"{x:.8f} {point_y:.8f} 2" for x, point_y in points)
    return f"{class_id} 0.50000000 {y + 0.0125:.8f} 0.16000000 0.02500000 {payload}"


def label(extra_classes=()) -> str:
    rows = [row(class_id, 0.03 + class_id * 0.04) for class_id in range(18)]
    rows.extend(row(class_id, 0.78 + index * 0.04) for index, class_id in enumerate(extra_classes))
    return "\n".join(rows) + "\n"


class RestoreCornerExtraClassesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source_labels = root / "source" / "labels"
        self.source_images = root / "source" / "images"
        self.corner = root / "corner"
        self.pose_roi = root / "pose_roi"
        self.corner_roi = root / "corner_roi"
        for dataset in (self.source_labels, self.source_images):
            for split in MODULE.SPLITS:
                (dataset / split).mkdir(parents=True)
        for split in MODULE.SPLITS:
            (self.corner / "images" / split).mkdir(parents=True)
            (self.corner / "labels" / split).mkdir(parents=True)
        (self.pose_roi / "images" / "train").mkdir(parents=True)
        (self.pose_roi / "manifest.json").write_text(json.dumps({"records": []}), encoding="utf-8")

        self.add_sample("sample_l6", "train", "train", (18,))
        self.add_sample("sample_t13", "test", "val", (19,))
        records, configuration = ROI.plan_views(self.corner, self.pose_roi)
        ROI.apply_plan(records, configuration, self.corner, self.pose_roi, self.corner_roi)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_sample(self, name: str, source_split: str, active_split: str, extras: tuple[int, ...]) -> None:
        source_image = self.source_images / source_split / f"{name}.png"
        active_image = self.corner / "images" / active_split / f"{name}.png"
        Image.new("L", (200, 400), 40 + len(name)).save(source_image)
        shutil.copy2(source_image, active_image)
        (self.source_labels / source_split / f"{name}.txt").write_text(label(extras), encoding="utf-8")
        (self.corner / "labels" / active_split / f"{name}.txt").write_text(label(), encoding="utf-8")

    def plan(self):
        return MODULE.build_restore_plan(
            self.source_labels,
            self.source_images,
            self.corner,
            expected_labels=2,
            expected_files=2,
            expected_class18=1,
            expected_class19=1,
        )

    def test_plan_requires_exact_base_payload_and_reports_current_split(self) -> None:
        items = self.plan()
        summary = MODULE.summarize_plan(items, MODULE.roi_preflight(items, self.corner_roi))
        self.assertEqual(summary["changed_raw_labels"], 2)
        self.assertEqual(summary["class_rows"], {"18": 1, "19": 1})
        self.assertEqual(summary["active_split_files"], {"train": 1, "val": 1, "test": 0})
        self.assertEqual(summary["roi"]["affected_train_files"], 1)

        source = self.source_labels / "train" / "sample_l6.txt"
        text = source.read_text(encoding="utf-8").replace("0.42000000 0.03000000", "0.41000000 0.03000000", 1)
        source.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "base keypoint payload mismatch"):
            self.plan()

    def test_apply_restores_raw_and_train_roi_labels_with_manifest(self) -> None:
        items = self.plan()
        summary = MODULE.summarize_plan(items, MODULE.roi_preflight(items, self.corner_roi))
        raw_image_hashes = {item.stem: MODULE.sha256_file(item.active_image) for item in items}
        backup = Path(self.temp.name) / "backup"
        record = Path(self.temp.name) / "record"
        result = MODULE.apply_restoration(
            items,
            summary,
            self.corner,
            self.pose_roi,
            self.corner_roi,
            backup,
            record,
            expected_labels=2,
            expected_roi_labels=1,
            expected_class18=1,
            expected_class19=1,
        )
        self.assertEqual(result["status"], "restored")
        self.assertEqual({item.class_id for item in MODULE.parse_label(self.corner / "labels/train/sample_l6.txt")}, set(range(19)))
        self.assertIn(19, {item.class_id for item in MODULE.parse_label(self.corner / "labels/val/sample_t13.txt")})
        roi_classes = {item.class_id for item in MODULE.parse_label(self.corner_roi / "labels/train/roi_sample_l6.txt")}
        self.assertIn(18, roi_classes)
        self.assertNotIn(19, roi_classes)
        self.assertTrue((backup / "manifest.json").is_file())
        self.assertTrue((record / "manifest.json").is_file())
        self.assertEqual(
            raw_image_hashes,
            {item.stem: MODULE.sha256_file(item.active_image) for item in items},
        )

    def test_failure_rolls_back_raw_roi_and_metadata(self) -> None:
        items = self.plan()
        summary = MODULE.summarize_plan(items, MODULE.roi_preflight(items, self.corner_roi))
        raw_before = {item.stem: MODULE.sha256_file(item.active_label) for item in items}
        roi_label = self.corner_roi / "labels/train/roi_sample_l6.txt"
        roi_before = MODULE.sha256_file(roi_label)
        metadata_before = MODULE.sha256_file(self.corner_roi / "manifest.json")
        backup = Path(self.temp.name) / "rollback_backup"
        with self.assertRaisesRegex(RuntimeError, "ROI source count mismatch"):
            MODULE.apply_restoration(
                items,
                summary,
                self.corner,
                self.pose_roi,
                self.corner_roi,
                backup,
                Path(self.temp.name) / "unused_record",
                expected_labels=2,
                expected_roi_labels=2,
                expected_class18=1,
                expected_class19=1,
            )
        self.assertEqual(raw_before, {item.stem: MODULE.sha256_file(item.active_label) for item in items})
        self.assertEqual(roi_before, MODULE.sha256_file(roi_label))
        self.assertEqual(metadata_before, MODULE.sha256_file(self.corner_roi / "manifest.json"))
        backup_manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(backup_manifest["status"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
