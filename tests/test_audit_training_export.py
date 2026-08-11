from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_training_export.py"
SPEC = importlib.util.spec_from_file_location("audit_training_export", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_png(path: Path, width: int = 100, height: int = 200, marker: bytes = b"") -> None:
    path.write_bytes(
        audit.PNG_SIGNATURE
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x00\x00\x00\x00"
        + marker
    )


def write_jpeg(path: Path, width: int = 100, height: int = 200) -> None:
    path.write_bytes(
        audit.JPEG_SIGNATURE
        + b"\xff\xe0"
        + struct.pack(">H", 4)
        + b"xx"
        + b"\xff\xc0"
        + struct.pack(">H", 7)
        + b"\x08"
        + struct.pack(">HH", height, width)
    )


def annotation(image_id: int, *, x: float = 0.2, width: int = 100) -> dict:
    return {
        "imageId": image_id,
        "originalFilename": "source.jpg",
        "imageWidth": width,
        "imageHeight": 200,
        "vertebrae": [
            {"label": "CL", "type": "point", "source": "manual", "point": {"x": x, "y": 0.2}},
            {"label": "CR", "type": "point", "source": "manual", "point": {"x": 0.8, "y": 0.2}},
        ],
    }


def write_pair(root: Path, image_id: int, data: dict, *, marker: bytes = b"") -> None:
    stem = f"{image_id}_case"
    write_png(root / f"{stem}.png", marker=marker)
    (root / f"{stem}_label.json").write_text(json.dumps(data), encoding="utf-8")


class AuditTrainingExportTests(unittest.TestCase):
    def test_reads_png_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            write_png(path, 321, 654)
            self.assertEqual(audit.read_image_dimensions(path), ("png", 321, 654))

    def test_reads_jpeg_dimensions_even_with_png_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            write_jpeg(path, 321, 654)
            self.assertEqual(audit.read_image_dimensions(path), ("jpeg", 321, 654))

    def test_reports_orphan_and_coordinate_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pair(root, 1, annotation(1, x=1.2))
            write_png(root / "2_orphan.png", marker=b"different")

            report = audit.audit_directory(root, hash_images=False)

            self.assertEqual(report["summary"]["images_without_annotations"], 1)
            self.assertEqual(report["statistics"]["issue_code_counts"]["coordinate_out_of_range"], 1)
            self.assertEqual(report["statistics"]["issue_code_counts"]["missing_annotation"], 1)

    def test_classifies_identical_and_conflicting_duplicate_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pair(root, 1, annotation(1))
            write_pair(root, 2, annotation(2))

            report = audit.audit_directory(root)
            self.assertEqual(report["summary"]["exact_duplicate_groups"], 1)
            self.assertEqual(report["exact_duplicate_groups"][0]["annotation_status"], "identical")

            second = root / "2_case_label.json"
            changed = annotation(2)
            changed["vertebrae"][0]["point"]["x"] = 0.3
            second.write_text(json.dumps(changed), encoding="utf-8")
            report = audit.audit_directory(root)
            self.assertEqual(report["exact_duplicate_groups"][0]["annotation_status"], "conflicting")
            self.assertEqual(
                report["exact_duplicate_groups"][0]["conflict_kind"],
                "coordinates_or_structure",
            )
            self.assertEqual(
                report["statistics"]["issue_code_counts"]["duplicate_image_conflicting_annotations"],
                1,
            )

    def test_reports_dimension_mismatch_and_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pair(root, 1, annotation(1, width=99))
            write_png(root / "2_case.png", marker=b"different")
            (root / "2_case_label.json").write_text("{broken", encoding="utf-8")

            report = audit.audit_directory(root, hash_images=False)
            codes = report["statistics"]["issue_code_counts"]
            self.assertEqual(codes["image_dimension_mismatch"], 1)
            self.assertEqual(codes["invalid_json"], 1)


if __name__ == "__main__":
    unittest.main()
