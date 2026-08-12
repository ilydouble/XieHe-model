from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "normalize_pose_lr_labels.py"
SPEC = importlib.util.spec_from_file_location("normalize_pose_lr_labels", SCRIPT)
assert SPEC and SPEC.loader
normalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalizer)


def label(xs: tuple[float, float, float, float, float, float]) -> str:
    points = " ".join(f"{x} 0.5 1" for x in xs)
    return f"0 0.5 0.5 0.8 0.8 {points}\n"


class NormalizePoseLrLabelsTests(unittest.TestCase):
    def test_apply_swaps_only_three_keypoint_triplets_and_backs_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "pose"
            source = dataset / "labels" / "train" / "case.txt"
            image = dataset / "images" / "train" / "case.png"
            source.parent.mkdir(parents=True)
            image.parent.mkdir(parents=True)
            original = label((0.2, 0.8, 0.25, 0.75, 0.4, 0.6))
            source.write_text(original, encoding="utf-8")
            image.write_bytes(b"unchanged image bytes")

            dry = normalizer.normalize_dataset(dataset)
            self.assertEqual(dry["patterns_before"], {"legacy_R_on_screen_left": 1})
            self.assertEqual(dry["changed_labels"], 1)

            backup = root / "backup"
            applied = normalizer.normalize_dataset(dataset, backup, apply=True)
            self.assertEqual(applied["patterns_after"], {"normalized_L_on_screen_left": 1})
            self.assertEqual((backup / "labels" / "train" / "case.txt").read_text(), original)
            self.assertEqual(image.read_bytes(), b"unchanged image bytes")
            tokens = source.read_text().split()
            self.assertEqual(tokens[:5], ["0", "0.5", "0.5", "0.8", "0.8"])
            self.assertEqual(
                [tokens[index] for index in (5, 8, 11, 14, 17, 20)],
                ["0.8", "0.2", "0.75", "0.25", "0.6", "0.4"],
            )

            again = normalizer.normalize_dataset(dataset, root / "unused", apply=True)
            self.assertEqual(again["status"], "already_normalized")

    def test_mixed_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            path = dataset / "labels" / "train" / "mixed.txt"
            path.parent.mkdir(parents=True)
            path.write_text(label((0.2, 0.8, 0.75, 0.25, 0.4, 0.6)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed"):
                normalizer.normalize_dataset(dataset)

    def test_syncs_only_detection_labels_with_matching_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / "pose"
            detection = root / "detection"
            pose_label = pose / "labels" / "test" / "matched.txt"
            matched = detection / "labels" / "test" / "matched.txt"
            orphan = detection / "labels" / "test" / "orphan.txt"
            pose_label.parent.mkdir(parents=True)
            matched.parent.mkdir(parents=True)
            pose_label.write_text(label((0.8, 0.2, 0.75, 0.25, 0.6, 0.4)), encoding="utf-8")
            matched.write_text("old\n", encoding="utf-8")
            orphan.write_text("orphan\n", encoding="utf-8")

            backup = root / "backup"
            result = normalizer.sync_derived_detection(pose, detection, backup, apply=True)

            self.assertEqual(result["matched_labels"], 1)
            self.assertEqual(result["orphan_labels_untouched"], 1)
            self.assertTrue(matched.read_text().startswith("0 0.800000 0.500000"))
            self.assertEqual(orphan.read_text(), "orphan\n")
            self.assertEqual(
                (backup / "detection_labels" / "test" / "matched.txt").read_text(),
                "old\n",
            )


if __name__ == "__main__":
    unittest.main()
