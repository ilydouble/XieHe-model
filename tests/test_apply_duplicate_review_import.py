from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"; sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("apply_duplicate_review_import", SCRIPTS / "apply_duplicate_review_import.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class ApplyReviewImportTests(unittest.TestCase):
    def test_audit_counts_pairs_and_detects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            line = "0 0.5 0.5 0.2 0.2 " + " ".join(["0.4 0.4 2"] * 6) + "\n"
            for split, name in (("train", "a"), ("val", "b")):
                image = root / "images" / split / f"{name}.png"; label = root / "labels" / split / f"{name}.txt"
                image.parent.mkdir(parents=True); label.parent.mkdir(parents=True)
                Image.new("L", (10, 10), 120).save(image); label.write_text(line)
            audit = MODULE.audit_dataset(root, "six_point")
            self.assertEqual(audit["total"], 2)
            self.assertEqual(len(audit["cross_split_exact_duplicate_groups"]), 1)

    def test_audit_rejects_reversed_six_point_lr_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image = root / "images/train/a.png"; label = root / "labels/train/a.txt"
            image.parent.mkdir(parents=True); label.parent.mkdir(parents=True)
            Image.new("L", (10, 10), 120).save(image)
            label.write_text("0 0.5 0.5 0.2 0.2 " + " ".join(["0.4 0.4 2", "0.6 0.4 2", "0.4 0.6 2", "0.6 0.6 2", "0.4 0.8 2", "0.6 0.8 2"]) + "\n")
            self.assertEqual(len(MODULE.audit_dataset(root, "six_point")["issues"]), 1)


if __name__ == "__main__": unittest.main()
