#!/usr/bin/env python3
"""Audit exported spine images and JSON annotations without modifying the dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABEL_SUFFIX = "_label.json"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"
POINT_LABELS = ("CL", "CR", "IL", "IR", "SL", "SR")
VERTEBRA_NAMES = ("C7",) + tuple(f"T{i}" for i in range(1, 13)) + tuple(
    f"L{i}" for i in range(1, 6)
)
VERTEBRA_LABELS = tuple(
    f"{vertebra}-{corner}"
    for vertebra in VERTEBRA_NAMES
    for corner in range(1, 5)
)
EXPECTED_LABELS = frozenset((*POINT_LABELS, *VERTEBRA_LABELS))
VALID_SOURCES = frozenset(("ai", "manual"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 training_export 中的重复图像、配对关系和 JSON 标注质量。"
    )
    parser.add_argument("export_dir", type=Path, help="包含 PNG 与 *_label.json 的导出目录")
    parser.add_argument("--output", type=Path, help="可选：写入完整 JSON 报告")
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="跳过图像 SHA-256 重复检查（速度更快，但不报告内容重复）",
    )
    parser.add_argument(
        "--fail-on",
        choices=("never", "error", "warning"),
        default="never",
        help="控制退出码：never（默认）、error 或 warning",
    )
    return parser.parse_args(argv)


def issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    file: str,
    message: str,
    **details: Any,
) -> None:
    record: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "file": file,
        "message": message,
    }
    if details:
        record["details"] = details
    issues.append(record)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_dimensions(path: Path) -> tuple[str, int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
        if len(header) >= 24 and header[:8] == PNG_SIGNATURE and header[12:16] == b"IHDR":
            width, height = struct.unpack(">II", header[16:24])
            if width <= 0 or height <= 0:
                raise ValueError("PNG 尺寸必须为正数")
            return "png", width, height
        if header[:2] != JPEG_SIGNATURE:
            raise ValueError("文件内容既不是 PNG 也不是 JPEG")

        stream.seek(2)
        start_of_frame = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while True:
            byte = stream.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker_bytes = stream.read(1)
            while marker_bytes == b"\xff":
                marker_bytes = stream.read(1)
            if not marker_bytes:
                break
            marker = marker_bytes[0]
            if marker in {0x01, *range(0xD0, 0xDA)}:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                raise ValueError("JPEG 段长度无效")
            if marker in start_of_frame:
                frame = stream.read(5)
                if len(frame) != 5:
                    break
                height, width = struct.unpack(">HH", frame[1:5])
                if width <= 0 or height <= 0:
                    raise ValueError("JPEG 尺寸必须为正数")
                return "jpeg", width, height
            stream.seek(segment_length - 2, 1)
    raise ValueError("JPEG 中未找到有效的 SOF 尺寸段")


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_coordinate(
    value: object,
    *,
    label: str,
    field: str,
    annotation_name: str,
    issues: list[dict[str, Any]],
) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        issue(
            issues,
            "error",
            "invalid_coordinate",
            annotation_name,
            f"{label}.{field} 必须是包含 x/y 的对象",
        )
        return None
    x, y = value.get("x"), value.get("y")
    if not is_number(x) or not is_number(y):
        issue(
            issues,
            "error",
            "invalid_coordinate",
            annotation_name,
            f"{label}.{field} 的 x/y 必须是数值",
        )
        return None
    x_value, y_value = float(x), float(y)
    if not math.isfinite(x_value) or not math.isfinite(y_value):
        issue(
            issues,
            "error",
            "non_finite_coordinate",
            annotation_name,
            f"{label}.{field} 含 NaN 或无穷值",
        )
        return None
    if not 0.0 <= x_value <= 1.0 or not 0.0 <= y_value <= 1.0:
        issue(
            issues,
            "error",
            "coordinate_out_of_range",
            annotation_name,
            f"{label}.{field} 超出归一化坐标范围 [0,1]",
            x=x_value,
            y=y_value,
        )
    return x_value, y_value


def normalized_annotation(data: dict[str, Any], *, ignore_sources: bool = False) -> str:
    items = data.get("vertebrae")
    normalized_items = items
    if isinstance(items, list):
        if ignore_sources:
            normalized_items = [
                {key: value for key, value in item.items() if key != "source"}
                if isinstance(item, dict)
                else item
                for item in items
            ]
        normalized_items = sorted(
            normalized_items,
            key=lambda item: str(item.get("label", "")) if isinstance(item, dict) else "",
        )
    normalized = {
        "imageWidth": data.get("imageWidth"),
        "imageHeight": data.get("imageHeight"),
        "vertebrae": normalized_items,
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_geometry(
    coordinates: dict[str, tuple[float, float]],
    annotation_name: str,
    issues: list[dict[str, Any]],
) -> None:
    for left, right in (("CL", "CR"), ("IL", "IR"), ("SL", "SR")):
        if left in coordinates and right in coordinates:
            if coordinates[left][0] >= coordinates[right][0]:
                issue(
                    issues,
                    "warning",
                    "left_right_order_anomaly",
                    annotation_name,
                    f"{left} 的 x 不小于 {right}，请确认左右点是否标反",
                    left_x=coordinates[left][0],
                    right_x=coordinates[right][0],
                )

    for vertebra in VERTEBRA_NAMES:
        labels = tuple(f"{vertebra}-{corner}" for corner in range(1, 5))
        if not all(label in coordinates for label in labels):
            continue
        p1, p2, p3, p4 = (coordinates[label] for label in labels)
        problems: list[str] = []
        if p1[0] >= p2[0] or p3[0] >= p4[0]:
            problems.append("左右角点顺序")
        if (p1[1] + p2[1]) / 2 >= (p3[1] + p4[1]) / 2:
            problems.append("上下角点顺序")
        if problems:
            issue(
                issues,
                "warning",
                "corner_order_anomaly",
                annotation_name,
                f"{vertebra} 的{'、'.join(problems)}异常",
            )


def validate_annotation(
    path: Path,
    image_path: Path | None,
    image_dimensions: tuple[int, int] | None,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, Counter[str], Counter[str], Counter[str]]:
    name = path.name
    empty = Counter()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issue(issues, "error", "invalid_json", name, f"JSON 无法读取或解析：{exc}")
        return None, empty, empty, empty
    if not isinstance(data, dict):
        issue(issues, "error", "invalid_root", name, "JSON 顶层必须是对象")
        return None, empty, empty, empty

    required = ("imageId", "originalFilename", "imageWidth", "imageHeight", "vertebrae")
    missing_fields = [field for field in required if field not in data]
    if missing_fields:
        issue(
            issues,
            "error",
            "missing_fields",
            name,
            "缺少顶层必填字段",
            fields=missing_fields,
        )

    expected_id = name[: -len(LABEL_SUFFIX)].split("_", 1)[0]
    if expected_id.isdigit() and data.get("imageId") != int(expected_id):
        issue(
            issues,
            "warning",
            "image_id_mismatch",
            name,
            "imageId 与导出文件名前缀不一致",
            expected=int(expected_id),
            actual=data.get("imageId"),
        )

    width, height = data.get("imageWidth"), data.get("imageHeight")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        issue(issues, "error", "invalid_image_width", name, "imageWidth 必须是正整数")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        issue(issues, "error", "invalid_image_height", name, "imageHeight 必须是正整数")
    if image_path is None:
        issue(issues, "error", "missing_image", name, "标注没有对应的同 stem PNG")
    elif image_dimensions and (width, height) != image_dimensions:
        issue(
            issues,
            "error",
            "image_dimension_mismatch",
            name,
            "JSON 尺寸与 PNG IHDR 尺寸不一致",
            json_dimensions=[width, height],
            png_dimensions=list(image_dimensions),
        )

    items = data.get("vertebrae")
    if not isinstance(items, list):
        issue(issues, "error", "invalid_annotations", name, "vertebrae 必须是数组")
        return data, empty, empty, empty
    if not items:
        issue(issues, "warning", "empty_annotations", name, "标注数组为空")

    label_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    coordinates: dict[str, tuple[float, float]] = {}

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            issue(
                issues,
                "error",
                "invalid_annotation_item",
                name,
                f"vertebrae[{index}] 必须是对象",
            )
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label:
            issue(issues, "error", "invalid_label", name, f"vertebrae[{index}].label 无效")
            label = f"<invalid:{index}>"
        label_counts[label] += 1

        annotation_type = item.get("type")
        source = item.get("source")
        type_counts[str(annotation_type)] += 1
        source_counts[str(source)] += 1
        if source not in VALID_SOURCES:
            issue(
                issues,
                "error",
                "invalid_source",
                name,
                f"{label}.source 必须是 ai 或 manual",
                actual=source,
            )

        expected_type = "point" if label in POINT_LABELS else "vertebra"
        if label in EXPECTED_LABELS and annotation_type != expected_type:
            issue(
                issues,
                "error",
                "annotation_type_mismatch",
                name,
                f"{label}.type 应为 {expected_type}",
                actual=annotation_type,
            )

        if annotation_type == "point":
            coordinate = validate_coordinate(
                item.get("point"),
                label=label,
                field="point",
                annotation_name=name,
                issues=issues,
            )
            if coordinate:
                coordinates[label] = coordinate
        elif annotation_type == "vertebra":
            corners = item.get("corners")
            if not isinstance(corners, list) or len(corners) != 4:
                issue(
                    issues,
                    "error",
                    "invalid_corners",
                    name,
                    f"{label}.corners 必须恰好包含 4 个坐标",
                )
                continue
            parsed = [
                validate_coordinate(
                    corner,
                    label=label,
                    field=f"corners[{corner_index}]",
                    annotation_name=name,
                    issues=issues,
                )
                for corner_index, corner in enumerate(corners)
            ]
            valid = [coordinate for coordinate in parsed if coordinate is not None]
            if valid:
                coordinates[label] = valid[0]
            if len(valid) == 4 and len(set(valid)) != 1:
                issue(
                    issues,
                    "warning",
                    "inconsistent_repeated_corners",
                    name,
                    f"{label} 的 4 个 corners 不完全相同；当前导出格式通常重复存储单个角点",
                )
        else:
            issue(
                issues,
                "error",
                "invalid_annotation_type",
                name,
                f"{label}.type 必须是 point 或 vertebra",
                actual=annotation_type,
            )

    duplicates = sorted(label for label, count in label_counts.items() if count > 1)
    if duplicates:
        issue(
            issues,
            "error",
            "duplicate_labels",
            name,
            "同一标注文件中存在重复 label",
            labels=duplicates,
        )
    present = set(label_counts)
    missing_labels = sorted(EXPECTED_LABELS - present)
    unexpected_labels = sorted(present - EXPECTED_LABELS)
    if missing_labels:
        issue(
            issues,
            "warning",
            "missing_expected_labels",
            name,
            f"缺少 {len(missing_labels)} 个期望标签",
            labels=missing_labels,
        )
    if unexpected_labels:
        issue(
            issues,
            "warning",
            "unexpected_labels",
            name,
            f"存在 {len(unexpected_labels)} 个非标准标签",
            labels=unexpected_labels,
        )
    validate_geometry(coordinates, name, issues)
    return data, label_counts, type_counts, source_counts


def audit_directory(export_dir: Path, *, hash_images: bool = True) -> dict[str, Any]:
    export_dir = export_dir.expanduser().resolve()
    if not export_dir.is_dir():
        raise NotADirectoryError(f"导出目录不存在或不是目录：{export_dir}")

    images = sorted(path for path in export_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    annotations = sorted(export_dir.glob(f"*{LABEL_SUFFIX}"))
    image_by_name = {path.name: path for path in images}
    issues: list[dict[str, Any]] = []
    dimensions: dict[str, tuple[int, int] | None] = {}
    image_formats: Counter[str] = Counter()
    hashes: dict[str, str] = {}

    for image in images:
        try:
            image_format, width, height = read_image_dimensions(image)
            dimensions[image.name] = (width, height)
            image_formats[image_format] += 1
        except (OSError, ValueError) as exc:
            dimensions[image.name] = None
            image_formats["invalid"] += 1
            issue(issues, "error", "invalid_image", image.name, f"图像无法识别或读取：{exc}")
        if hash_images:
            try:
                hashes[image.name] = sha256_file(image)
            except OSError as exc:
                issue(issues, "error", "image_hash_failed", image.name, f"无法计算 SHA-256：{exc}")

    annotation_data: dict[str, dict[str, Any]] = {}
    annotation_for_image: dict[str, str] = {}
    all_label_counts: Counter[str] = Counter()
    all_type_counts: Counter[str] = Counter()
    all_source_counts: Counter[str] = Counter()
    annotation_count_distribution: Counter[int] = Counter()
    original_filenames: defaultdict[str, list[str]] = defaultdict(list)

    for annotation in annotations:
        image_name = f"{annotation.name[: -len(LABEL_SUFFIX)]}.png"
        image_path = image_by_name.get(image_name)
        data, labels, types, sources = validate_annotation(
            annotation,
            image_path,
            dimensions.get(image_name),
            issues,
        )
        all_label_counts.update(labels)
        all_type_counts.update(types)
        all_source_counts.update(sources)
        if data is not None:
            annotation_data[annotation.name] = data
            annotation_for_image[image_name] = annotation.name
            items = data.get("vertebrae")
            if isinstance(items, list):
                annotation_count_distribution[len(items)] += 1
            original = data.get("originalFilename")
            if isinstance(original, str):
                original_filenames[original].append(annotation.name)

    orphan_images = sorted(set(image_by_name) - set(annotation_for_image))
    for image_name in orphan_images:
        issue(issues, "warning", "missing_annotation", image_name, "PNG 没有对应的 *_label.json")

    duplicate_groups: list[dict[str, Any]] = []
    if hash_images:
        grouped_hashes: defaultdict[str, list[str]] = defaultdict(list)
        for image_name, digest in hashes.items():
            grouped_hashes[digest].append(image_name)
        for digest, names in sorted(grouped_hashes.items()):
            if len(names) < 2:
                continue
            annotation_hashes: defaultdict[str, list[str]] = defaultdict(list)
            geometry_hashes: set[str] = set()
            for image_name in sorted(names):
                annotation_name = annotation_for_image.get(image_name)
                if annotation_name and annotation_name in annotation_data:
                    normalized = normalized_annotation(annotation_data[annotation_name])
                    annotation_hashes[hashlib.sha256(normalized.encode("utf-8")).hexdigest()].append(
                        annotation_name
                    )
                    geometry = normalized_annotation(
                        annotation_data[annotation_name], ignore_sources=True
                    )
                    geometry_hashes.add(hashlib.sha256(geometry.encode("utf-8")).hexdigest())
            labeled_count = sum(len(group) for group in annotation_hashes.values())
            if not annotation_hashes:
                status = "unlabeled"
            elif labeled_count < len(names):
                status = "partial"
            elif len(annotation_hashes) == 1:
                status = "identical"
            else:
                status = "conflicting"
                issue(
                    issues,
                    "warning",
                    "duplicate_image_conflicting_annotations",
                    ", ".join(sorted(names)),
                    "完全相同的图像对应不同标注内容，去重前需要人工选择或合并",
                    annotation_variants=len(annotation_hashes),
                )
            if status != "conflicting":
                conflict_kind = "none"
            elif len(geometry_hashes) == 1:
                conflict_kind = "source_only"
            else:
                conflict_kind = "coordinates_or_structure"
            duplicate_groups.append(
                {
                    "sha256": digest,
                    "files": sorted(names),
                    "annotation_status": status,
                    "conflict_kind": conflict_kind,
                    "annotation_variants": [sorted(group) for group in annotation_hashes.values()],
                }
            )

    duplicate_original_filenames = [
        {"original_filename": original, "annotations": sorted(names)}
        for original, names in sorted(original_filenames.items())
        if len(names) > 1
    ]
    issue_counts = Counter(record["severity"] for record in issues)
    code_counts = Counter(record["code"] for record in issues)
    duplicate_status_counts = Counter(group["annotation_status"] for group in duplicate_groups)
    conflict_kind_counts = Counter(group["conflict_kind"] for group in duplicate_groups)
    redundant_images = sum(len(group["files"]) - 1 for group in duplicate_groups)

    return {
        "schema_version": 1,
        "export_dir": str(export_dir),
        "summary": {
            "images": len(images),
            "annotations": len(annotations),
            "paired": len(annotation_for_image),
            "images_without_annotations": len(orphan_images),
            "exact_duplicate_groups": len(duplicate_groups),
            "images_in_exact_duplicate_groups": sum(len(group["files"]) for group in duplicate_groups),
            "redundant_exact_images": redundant_images,
            "duplicate_annotation_status": dict(sorted(duplicate_status_counts.items())),
            "duplicate_conflict_kind": dict(sorted(conflict_kind_counts.items())),
            "duplicate_original_filename_groups": len(duplicate_original_filenames),
            "errors": issue_counts["error"],
            "warnings": issue_counts["warning"],
        },
        "statistics": {
            "annotation_count_distribution": {
                str(key): value for key, value in sorted(annotation_count_distribution.items())
            },
            "image_formats": dict(sorted(image_formats.items())),
            "type_counts": dict(sorted(all_type_counts.items())),
            "source_counts": dict(sorted(all_source_counts.items())),
            "label_counts": dict(sorted(all_label_counts.items())),
            "issue_code_counts": dict(sorted(code_counts.items())),
        },
        "exact_duplicate_groups": duplicate_groups,
        "duplicate_original_filenames": duplicate_original_filenames,
        "issues": issues,
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"图像: {summary['images']}")
    print(f"标注: {summary['annotations']}（配对 {summary['paired']}）")
    print(f"无标注图像: {summary['images_without_annotations']}")
    print(
        "精确重复: "
        f"{summary['exact_duplicate_groups']} 组，"
        f"涉及 {summary['images_in_exact_duplicate_groups']} 张，"
        f"冗余 {summary['redundant_exact_images']} 张"
    )
    statuses = summary["duplicate_annotation_status"]
    if statuses:
        rendered = "，".join(f"{key}={value}" for key, value in statuses.items())
        print(f"重复图像标注状态: {rendered}")
    conflict_kinds = summary["duplicate_conflict_kind"]
    if conflict_kinds:
        rendered = "，".join(f"{key}={value}" for key, value in conflict_kinds.items())
        print(f"标注冲突类型: {rendered}")
    print(f"问题: error={summary['errors']}，warning={summary['warnings']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_directory(args.export_dir, hash_images=not args.skip_hashes)
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print_summary(report)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"完整报告: {output}")

    summary = report["summary"]
    if args.fail_on == "error" and summary["errors"]:
        return 1
    if args.fail_on == "warning" and (summary["errors"] or summary["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
