from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "normalize_corner_bboxes.py"
SPEC = importlib.util.spec_from_file_location("normalize_corner_bboxes", SCRIPT)
assert SPEC and SPEC.loader
normalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalizer)


def row(class_id: int, bbox: str = "0.5 0.5 0.1 0.1") -> str:
    return (
        f"{class_id} {bbox} "
        "0.20 0.30 2 0.80 0.35 2 0.75 0.70 2 0.25 0.65 2"
    )


class NormalizeCornerBboxesTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> tuple[Path, Path, str]:
        dataset = root / "corner"
        label = dataset / "labels" / "train" / "case.txt"
        image = dataset / "images" / "train" / "case.png"
        label.parent.mkdir(parents=True)
        image.parent.mkdir(parents=True)
        original = "\n".join(row(class_id) for class_id in range(18)) + "\n"
        label.write_text(original, encoding="utf-8")
        image.write_bytes(b"unchanged image")
        return dataset, label, original

    def test_apply_changes_only_bbox_and_creates_full_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, label, original = self.make_dataset(root)
            dry = normalizer.normalize_dataset(dataset, expected_labels=1)
            self.assertEqual(dry["changed_files"], 1)
            self.assertEqual(dry["changed_rows"], 18)

            backup = root / "backup"
            result = normalizer.normalize_dataset(dataset, backup, apply=True, expected_labels=1)
            self.assertEqual(result["status"], "normalized")
            self.assertEqual(result["audit"]["issues"], [])
            self.assertEqual((backup / "labels/train/case.txt").read_text(), original)
            self.assertEqual((dataset / "images/train/case.png").read_bytes(), b"unchanged image")

            old_rows = [line.split() for line in original.splitlines()]
            new_rows = [line.split() for line in label.read_text().splitlines()]
            for old, new in zip(old_rows, new_rows):
                self.assertEqual(old[0], new[0])
                self.assertEqual(old[5:], new[5:])
                self.assertEqual(new[1:5], ["0.50000000", "0.50000000", "0.60000000", "0.40000000"])

            again = normalizer.normalize_dataset(dataset, root / "unused", apply=True, expected_labels=1)
            self.assertEqual(again["status"], "already_normalized")

    def test_rejects_change_outside_bbox_or_bad_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, label, _ = self.make_dataset(root)
            label.write_text(row(0) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "缺失"):
                normalizer.normalize_dataset(dataset, expected_labels=1)

    def test_expected_label_count_is_a_safety_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, _, _ = self.make_dataset(Path(directory))
            with self.assertRaisesRegex(ValueError, "安全门槛"):
                normalizer.normalize_dataset(dataset, expected_labels=2)


if __name__ == "__main__":
    unittest.main()
