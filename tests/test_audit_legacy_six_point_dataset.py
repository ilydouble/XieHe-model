import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_legacy_six_point_dataset.py"
SPEC = importlib.util.spec_from_file_location("legacy_audit", SCRIPT)
legacy_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(legacy_audit)


def write_pair(root: Path, split: str, stem: str, label: str) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "images" / split / f"{stem}.png").write_bytes(f"image-{split}-{stem}".encode())
    (root / "labels" / split / f"{stem}.txt").write_text(label, encoding="utf-8")


class LegacySixPointAuditTest(unittest.TestCase):
    def test_classifies_mixed_pose_tasks_and_matches_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pose = root / "pose"
            detection = root / "detection"
            points = " ".join(f"{0.03 + i * 0.01:.3f} {0.04 + i * 0.01:.3f} 2" for i in range(6))
            pose_line = f"0 0.06 0.07 0.08 0.09 {points}\n"
            write_pair(pose, "train", "six", pose_line)
            corner_line = "1 0.5 0.5 0.1 0.1 0.4 0.4 2 0.6 0.4 2 0.6 0.6 2 0.4 0.6 2\n"
            write_pair(pose, "test", "corner", corner_line)

            values = [float(value) for value in pose_line.split()]
            det_lines = "\n".join(" ".join(f"{value:.6f}" for value in row) for row in legacy_audit.expected_detection(values)) + "\n"
            write_pair(detection, "train", "six", det_lines)

            result = legacy_audit.audit(pose, detection)
            self.assertEqual(result["pose_label_types"], {"six_point_pose": 1, "vertebra_corner_pose": 1})
            self.assertEqual(result["issue_reasons"]["all_visible_points_top_left"], 1)
            self.assertEqual(result["issue_reasons"]["vertebra_corner_label_in_six_point_dataset"], 1)
            self.assertEqual(result["detection"]["exact_pose_conversion_matches"], 1)
            self.assertEqual(result["detection"]["conversion_mismatches"], 0)

    def test_detects_cross_split_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            pose = Path(temp) / "pose"
            points = " ".join("0.3 0.4 1" for _ in range(6))
            label = f"0 0.5 0.5 0.5 0.5 {points}\n"
            write_pair(pose, "train", "a", label)
            write_pair(pose, "val", "b", label)
            (pose / "images" / "val" / "b.png").write_bytes((pose / "images" / "train" / "a.png").read_bytes())

            result = legacy_audit.audit(pose, None)
            self.assertEqual(len(result["exact_duplicate_groups"]), 1)
            self.assertEqual(len(result["cross_split_exact_duplicate_groups"]), 1)

    def test_flags_left_right_and_iliac_sacral_structure_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            pose = Path(temp) / "pose"
            points = (
                "0.2 0.2 1 "  # CR is incorrectly to the left of CL.
                "0.8 0.2 1 "
                "0.7 0.7 1 "
                "0.3 0.7 1 "
                "0.6 0.6 1 "  # Sacral pair is implausibly above iliac pair.
                "0.4 0.6 1"
            )
            write_pair(pose, "train", "bad_structure", f"0 0.5 0.45 0.6 0.5 {points}\n")

            result = legacy_audit.audit(pose, None)

            self.assertEqual(result["issue_reasons"]["clavicle_left_right_order_conflict"], 1)
            self.assertEqual(result["issue_reasons"]["sacral_pair_above_iliac_pair"], 1)
            self.assertEqual(result["split_six_point_quality"]["train"]["needs_review"], 1)

    def test_counts_duplicate_images_with_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            pose = Path(temp) / "pose"
            points_a = " ".join("0.3 0.4 1" for _ in range(6))
            points_b = " ".join("0.4 0.4 1" for _ in range(6))
            write_pair(pose, "train", "a", f"0 0.5 0.5 0.5 0.5 {points_a}\n")
            write_pair(pose, "val", "b", f"0 0.5 0.5 0.5 0.5 {points_b}\n")
            (pose / "images" / "val" / "b.png").write_bytes((pose / "images" / "train" / "a.png").read_bytes())

            result = legacy_audit.audit(pose, None)

            self.assertEqual(result["exact_duplicate_label_conflict_groups"], 1)


if __name__ == "__main__":
    unittest.main()
