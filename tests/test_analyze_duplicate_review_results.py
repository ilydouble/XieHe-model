from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_duplicate_review_results.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("analyze_duplicate_review_results", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReviewResultAnalysisTests(unittest.TestCase):
    def test_standard_choice_applies_to_both_tasks(self) -> None:
        row = {"组号": "G1", "选择": "candidate:1", "备注": ""}
        self.assertEqual(MODULE.parse_task_choices(row, 2), {"six_point": 1, "spine_pose": 1})

    def test_split_note_selects_different_versions(self) -> None:
        row = {"组号": "G1", "选择": "neither", "备注": "椎体用图1，六点用图2"}
        self.assertEqual(MODULE.parse_task_choices(row, 2), {"six_point": 1, "spine_pose": 0})

    def test_unactionable_neither_remains_unselected(self) -> None:
        row = {"组号": "G1", "选择": "neither", "备注": "图2的CR修改后才行"}
        self.assertEqual(MODULE.parse_task_choices(row, 2), {"six_point": None, "spine_pose": None})

    def test_split_note_accepts_image_first_and_arabic_six(self) -> None:
        row = {"组号": "G1", "选择": "neither", "备注": "选择图1的6点，图2的椎体"}
        self.assertEqual(MODULE.parse_task_choices(row, 2), {"six_point": 0, "spine_pose": 1})


if __name__ == "__main__":
    unittest.main()
