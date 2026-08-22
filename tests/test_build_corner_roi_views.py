#!/usr/bin/env python3
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_corner_roi_views.py"
SPEC = importlib.util.spec_from_file_location("build_corner_roi_views", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildCornerRoiViewsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.corner = root / "pose_corner_data"
        self.pose_roi = root / "pose_roi_views"
        self.output = root / "corner_roi_views"
        (self.corner / "images" / "train").mkdir(parents=True)
        (self.corner / "labels" / "train").mkdir(parents=True)
        (self.pose_roi / "images" / "train").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def add_corner_sample(self, name: str, points=None):
        image_path = self.corner / "images" / "train" / f"{name}.png"
        label_path = self.corner / "labels" / "train" / f"{name}.txt"
        Image.new("L", (200, 400), 80).save(image_path)
        points = points or ((0.42, 0.25), (0.58, 0.25), (0.58, 0.30), (0.42, 0.30))
        lines = []
        for class_id in range(2):
            shifted = tuple((x, y + class_id * 0.35) for x, y in points)
            xs, ys = [p[0] for p in shifted], [p[1] for p in shifted]
            encoded = " ".join(f"{x} {y} 2" for x, y in shifted)
            lines.append(f"{class_id} {(min(xs)+max(xs))/2} {(min(ys)+max(ys))/2} {max(xs)-min(xs)} {max(ys)-min(ys)} {encoded}")
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return image_path, label_path

    def write_pose_manifest(self, samples):
        records = []
        for name, box in samples:
            source = self.corner / "images" / "train" / f"{name}.png"
            output = self.pose_roi / "images" / "train" / f"roi_{name}.png"
            with Image.open(source) as image:
                image.crop((box.left, box.top, box.right, box.bottom)).save(output)
            records.append(
                {
                    "source_image": f"images/train/{name}.png",
                    "source_width": 200,
                    "source_height": 400,
                    "source_image_sha256": MODULE.sha256_file(source),
                    "crop_box": MODULE.asdict(box),
                    "output_image": f"images/train/roi_{name}.png",
                    "output_image_sha256": MODULE.sha256_file(output),
                }
            )
        (self.pose_roi / "manifest.json").write_text(json.dumps({"records": records}), encoding="utf-8")

    def test_transform_round_trip_geometry(self):
        _, label_path = self.add_corner_sample("sample")
        objects = MODULE.parse_corner_label(label_path)
        box = MODULE.CropBox(20, 40, 180, 360)
        changed = MODULE.transform_objects(objects, box, 200, 400)
        for original, transformed in zip(objects, changed):
            for before, after in zip(original.keypoints, transformed.keypoints):
                self.assertAlmostEqual(before[0] * 200, after[0] * box.width + box.left)
                self.assertAlmostEqual(before[1] * 400, after[1] * box.height + box.top)

    def test_safe_pose_roi_is_hardlinked_with_corner_labels(self):
        self.add_corner_sample("safe")
        self.write_pose_manifest([("safe", MODULE.CropBox(20, 40, 180, 360))])
        records, config = MODULE.plan_views(self.corner, self.pose_roi)
        self.assertEqual(records[0]["plan_reason"], "reused_pose_roi")
        manifest = MODULE.apply_plan(records, config, self.corner, self.pose_roi, self.output)
        output_image = self.output / records[0]["output_image"]
        pose_image = self.pose_roi / records[0]["reuse_image"]
        self.assertTrue(os.path.samefile(output_image, pose_image))
        output_objects = MODULE.parse_corner_label(self.output / records[0]["output_label"])
        self.assertEqual(len(output_objects), 2)
        self.assertEqual(manifest["summary"]["reused_pose_roi_hardlink_count"], 1)

    def test_unsafe_pose_roi_is_expanded_and_regenerated(self):
        self.add_corner_sample("unsafe")
        self.write_pose_manifest([("unsafe", MODULE.CropBox(70, 120, 130, 250))])
        records, config = MODULE.plan_views(self.corner, self.pose_roi)
        self.assertEqual(records[0]["plan_reason"], "expanded_unsafe_pose_roi")
        MODULE.apply_plan(records, config, self.corner, self.pose_roi, self.output)
        output_image = self.output / records[0]["output_image"]
        pose_image = self.pose_roi / "images/train/roi_unsafe.png"
        self.assertFalse(os.path.samefile(output_image, pose_image))
        self.assertTrue(MODULE.crop_contains_target(MODULE.CropBox(**records[0]["crop_box"]), MODULE.parse_corner_label(self.corner / records[0]["source_label"]), 200, 400))

    def test_missing_pose_roi_is_generated_deterministically(self):
        self.add_corner_sample("missing")
        self.write_pose_manifest([])
        first, config = MODULE.plan_views(self.corner, self.pose_roi)
        second, _ = MODULE.plan_views(self.corner, self.pose_roi)
        self.assertEqual(first[0]["crop_box"], second[0]["crop_box"])
        self.assertEqual(first[0]["plan_reason"], "missing_pose_roi")
        manifest = MODULE.apply_plan(first, config, self.corner, self.pose_roi, self.output)
        self.assertEqual(manifest["summary"]["new_pixel_file_count"], 1)

    def test_second_apply_skips_unchanged_outputs(self):
        self.add_corner_sample("safe")
        self.write_pose_manifest([("safe", MODULE.CropBox(20, 40, 180, 360))])
        records, config = MODULE.plan_views(self.corner, self.pose_roi)
        MODULE.apply_plan(records, config, self.corner, self.pose_roi, self.output)
        inode = (self.output / records[0]["output_image"]).stat().st_ino
        records, config = MODULE.plan_views(self.corner, self.pose_roi)
        manifest = MODULE.apply_plan(records, config, self.corner, self.pose_roi, self.output)
        self.assertEqual(manifest["summary"]["skipped_existing_count"], 1)
        self.assertEqual((self.output / records[0]["output_image"]).stat().st_ino, inode)


if __name__ == "__main__":
    unittest.main()
