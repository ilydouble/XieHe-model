from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "resplit_pose_data_by_patient.py"
SPEC = importlib.util.spec_from_file_location("resplit_pose", SCRIPT)
assert SPEC and SPEC.loader
resplit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resplit)


class ResplitPoseDataTests(unittest.TestCase):
    def test_selects_whole_patient_groups_with_exact_count(self) -> None:
        groups = {"p1": [{}, {}], "p2": [{}], "p3": [{}, {}], "p4": [{}]}
        selected = resplit.choose_groups(groups, 3, seed=7)
        self.assertEqual(sum(len(groups[key]) for key in selected), 3)

    def test_apply_moves_pairs_and_preserves_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pose"
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            actions = []
            for index in range(10):
                stem = f"eap_{index}"
                image = root / "images" / "train" / f"{stem}.png"
                label = root / "labels" / "train" / f"{stem}.txt"
                image.write_bytes(f"image-{index}".encode())
                label.write_text(f"label-{index}", encoding="utf-8")
                actions.append({
                    "status": "imported",
                    "assignment_patient_id": f"p{index}",
                    "source_image": f"{index}.png",
                    "destination_image": str(image),
                    "destination_label": str(label),
                })
            manifest = Path(directory) / "import.json"
            manifest.write_text(json.dumps({"actions": {"six_point": actions}}), encoding="utf-8")
            plan = resplit.build_plan(root, manifest)
            self.assertEqual(plan["target_counts"], {"train": 8, "val": 1, "test": 1})
            resplit.apply_plan(plan)
            self.assertEqual(resplit.split_counts(root), plan["target_counts"])
            self.assertTrue(all(move["status"] == "moved" for move in plan["moves"]))


if __name__ == "__main__":
    unittest.main()
