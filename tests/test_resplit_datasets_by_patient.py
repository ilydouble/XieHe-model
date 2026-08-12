from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"; sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("resplit_datasets_by_patient", SCRIPTS / "resplit_datasets_by_patient.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class PatientResplitTests(unittest.TestCase):
    def test_choose_exact_prefers_cross_task_groups(self) -> None:
        groups = [("a", 2), ("b", 2), ("c", 1)]
        self.assertEqual(MODULE.choose_exact(groups, 3, {"a"}), {"a", "c"})

    def test_patient_key_parses_supported_schemes(self) -> None:
        ids = {"SCO2003P1027"}
        self.assertEqual(MODULE.patient_key("eap_1_SCO2003P1027_20160714.png", ids)[0], "assignment:SCO2003P1027")
        self.assertEqual(MODULE.patient_key("1234567890__CR_TSPINE_20200101_slice0000.png", ids)[0], "server:1234567890")
        self.assertFalse(MODULE.patient_key("1.2.156.147522.44.410947.985.1.1.20250526153314.png", ids)[1])
        self.assertTrue(MODULE.patient_key("unknown.png", ids)[0].startswith("unparsed:"))


if __name__ == "__main__": unittest.main()
