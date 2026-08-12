from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_corner_dataset_with_server.py"
SPEC = importlib.util.spec_from_file_location("rebuild_corner_dataset_with_server", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def label(extra: bool = False, missing12: bool = False) -> str:
    rows = []
    for class_id in range(18):
        if missing12 and class_id == 12:
            continue
        rows.append(f"{class_id} 0.5 0.5 0.2 0.1 0.4 0.4 2 0.6 0.4 2 0.6 0.6 2 0.4 0.6 2")
    if extra:
        rows.append("18 0.5 0.5 0.2 0.1 0.4 0.4 2 0.6 0.4 2 0.6 0.6 2 0.4 0.6 2")
    return "\n".join(rows) + "\n"


def sample(root: Path, split: str, name: str, content: str) -> None:
    image = root / "images" / split / f"{name}.png"
    target = root / "labels" / split / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (20, 30), len(name) * 10).save(image)
    target.write_text(content, encoding="utf-8")


class RebuildCornerDatasetTests(unittest.TestCase):
    def test_plan_and_apply_replace_legacy_keep_eap_and_filter_extra_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, server = root / "target", root / "server"
            sample(target, "train", "legacy", label())
            sample(target, "train", "eap_human", label())
            sample(server, "train", "server_a", label(extra=True))
            sample(server, "val", "server_b", label(missing12=True))
            plan = MODULE.build_plan(
                server, target, expected_legacy=1, expected_retained=1, expected_server=2
            )
            self.assertEqual(plan["expected_after"]["split_counts"], {"train": 2, "val": 1, "test": 0})
            self.assertEqual(plan["server"]["files_with_removed_class_18_19"], 1)
            self.assertEqual(plan["server"]["files_missing_class_12"], 1)
            quarantine = root / "quarantine"
            MODULE.apply_plan(plan, target, quarantine)
            self.assertFalse((target / "images/train/legacy.png").exists())
            self.assertTrue((quarantine / "images/train/legacy.png").is_file())
            self.assertTrue((target / "images/train/eap_human.png").is_file())
            imported = (target / "labels/train/server_a.txt").read_text(encoding="utf-8")
            self.assertNotIn("\n18 ", "\n" + imported)
            self.assertEqual(len(MODULE.paired_files(target)), 3)
            audit = MODULE.audit_dataset(target)
            self.assertEqual(audit["issues"], [])
            self.assertEqual(audit["missing_class_12"], ["val/server_b.png"])
            self.assertEqual(MODULE.verify_quarantine(plan, quarantine)["issues"], [])

    def test_rejects_retained_filename_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, server = root / "target", root / "server"
            sample(target, "train", "eap_same", label())
            sample(server, "val", "eap_same", label())
            with self.assertRaisesRegex(ValueError, "文件名.*冲突"):
                MODULE.build_plan(
                    server, target, expected_legacy=0, expected_retained=1, expected_server=1
                )

    def test_rejects_invalid_corner_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text("0 0.5 0.5 0.2 0.1 0.6 0.4 2 0.4 0.4 2 0.4 0.6 2 0.6 0.6 2\n")
            with self.assertRaisesRegex(ValueError, "几何顺序异常"):
                MODULE.sanitize_label(path)


if __name__ == "__main__":
    unittest.main()
