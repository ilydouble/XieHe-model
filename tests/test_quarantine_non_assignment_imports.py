from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "quarantine_non_assignment_imports", SCRIPTS / "quarantine_non_assignment_imports.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
from test_import_e_drive_training_data import write_assignment_xlsx  # noqa: E402


class QuarantineTests(unittest.TestCase):
    def test_builds_and_applies_recoverable_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignment = root / "assignment.xlsx"
            write_assignment_xlsx(assignment)
            image = root / "images" / "train" / "eap_test.png"
            label = root / "labels" / "train" / "eap_test.txt"
            image.parent.mkdir(parents=True); label.parent.mkdir(parents=True)
            image.write_bytes(b"image"); label.write_text("label", encoding="utf-8")
            manifest = root / "import.json"
            manifest.write_text(json.dumps({"actions": {"spine_pose": [{
                "status": "imported", "source_image": "test.png",
                "destination_image": str(image), "destination_label": str(label),
            }]}}), encoding="utf-8")
            plan = module.build_plan(manifest, assignment)
            self.assertEqual(len(plan["records"]), 1)
            quarantine = root / "quarantine"
            module.apply_plan(plan, quarantine)
            self.assertFalse(image.exists()); self.assertFalse(label.exists())
            self.assertTrue(all(Path(item["quarantine_path"]).is_file() for item in plan["records"][0]["files"]))


if __name__ == "__main__":
    unittest.main()
