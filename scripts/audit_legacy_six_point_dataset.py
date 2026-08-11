#!/usr/bin/env python3
"""Audit the legacy six-point Pose dataset and its derived Detection dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
KEYPOINT_NAMES = ("CR", "CL", "IR", "IL", "SR", "SL")
CSV_FIELDS = ("split", "label", "image", "kind", "decision", "reasons")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计旧六点Pose数据及其Detection派生数据。")
    parser.add_argument("pose_dir", type=Path)
    parser.add_argument("--detection-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-left-threshold", type=float, default=0.2)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[list[float]], str | None]:
    rows: list[list[float]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append([float(value) for value in line.split()])
    except (OSError, ValueError) as exc:
        return [], str(exc)
    return rows, None


def classify_pose(rows: list[list[float]]) -> str:
    if len(rows) == 1 and len(rows[0]) == 23 and int(rows[0][0]) == 0:
        return "six_point_pose"
    if rows and all(len(row) == 17 for row in rows):
        return "vertebra_corner_pose"
    return "invalid_or_unknown"


def image_map(root: Path, split: str) -> dict[str, Path]:
    directory = root / "images" / split
    if not directory.exists():
        return {}
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES
    }


def label_map(root: Path, split: str) -> dict[str, Path]:
    directory = root / "labels" / split
    if not directory.exists():
        return {}
    return {path.stem: path for path in directory.glob("*.txt") if path.is_file()}


def visible_points(values: list[float]) -> list[tuple[float, float, float]]:
    points = [tuple(values[5 + index * 3 : 8 + index * 3]) for index in range(6)]
    return [(x, y, visibility) for x, y, visibility in points if visibility > 0]


def six_point_reasons(values: list[float], threshold: float) -> list[str]:
    reasons: list[str] = []
    points = [tuple(values[5 + index * 3 : 8 + index * 3]) for index in range(6)]
    if any(not math.isfinite(value) for value in values):
        reasons.append("non_finite_value")
    if any(not 0 <= value <= 1 for value in values[1:5]):
        reasons.append("bbox_out_of_range")
    if any(not 0 <= coordinate <= 1 for x, y, _ in points for coordinate in (x, y)):
        reasons.append("keypoint_out_of_range")
    if any(visibility not in {0, 1, 2} for _, _, visibility in points):
        reasons.append("invalid_visibility")
    visible = [(x, y, visibility) for x, y, visibility in points if visibility > 0]
    if visible and all(x <= threshold and y <= threshold for x, y, _ in visible):
        reasons.append("all_visible_points_top_left")
    if visible:
        cx, cy, width, height = values[1:5]
        x1, x2 = cx - width / 2, cx + width / 2
        y1, y2 = cy - height / 2, cy + height / 2
        if any(not (x1 - 1e-5 <= x <= x2 + 1e-5 and y1 - 1e-5 <= y <= y2 + 1e-5) for x, y, _ in visible):
            reasons.append("bbox_does_not_enclose_visible_points")
    if len(visible) == 6:
        for name, right_index, left_index in (
            ("clavicle", 0, 1),
            ("iliac", 2, 3),
            ("sacral", 4, 5),
        ):
            if points[right_index][0] >= points[left_index][0]:
                reasons.append(f"{name}_left_right_order_conflict")
        iliac_y = (points[2][1] + points[3][1]) / 2
        sacral_y = (points[4][1] + points[5][1]) / 2
        if sacral_y < iliac_y - 0.01:
            reasons.append("sacral_pair_above_iliac_pair")
    return reasons


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_detection(values: list[float], bbox_ratio: float = 0.04) -> list[list[float]]:
    expected: list[list[float]] = []
    half = bbox_ratio / 2
    for index in range(6):
        x, y, visibility = values[5 + index * 3 : 8 + index * 3]
        if int(visibility) == 0:
            continue
        expected.append(
            [float(index), max(half, min(1 - half, x)), max(half, min(1 - half, y)), bbox_ratio, bbox_ratio]
        )
    return expected


def rows_close(left: list[list[float]], right: list[list[float]], tolerance: float = 5e-7) -> bool:
    return len(left) == len(right) and all(
        len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))
        for a, b in zip(left, right)
    )


def audit(pose_dir: Path, detection_dir: Path | None, threshold: float = 0.2) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    pose_counts: Counter[str] = Counter()
    split_label_types: dict[str, Counter[str]] = {}
    split_six_point_quality: dict[str, Counter[str]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    six_labels: dict[tuple[str, str], list[float]] = {}
    visibility: Counter[str] = Counter()
    left_right_patterns: Counter[str] = Counter()
    hashes: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for split in SPLITS:
        images = image_map(pose_dir, split)
        labels = label_map(pose_dir, split)
        split_counts[split] = {"images": len(images), "labels": len(labels)}
        split_label_types[split] = Counter()
        split_six_point_quality[split] = Counter()
        for stem in sorted(images.keys() - labels.keys()):
            issues.append({"split": split, "label": "", "image": images[stem].name, "kind": "pairing", "decision": "排除", "reasons": "image_without_label"})
        for stem in sorted(labels.keys() - images.keys()):
            issues.append({"split": split, "label": labels[stem].name, "image": "", "kind": "pairing", "decision": "排除", "reasons": "label_without_image"})
        for stem, image in images.items():
            hashes[sha256(image)].append({"split": split, "image": image.name})
        for stem, label in sorted(labels.items()):
            rows, error = read_rows(label)
            kind = classify_pose(rows) if error is None else "invalid_or_unknown"
            pose_counts[kind] += 1
            split_label_types[split][kind] += 1
            image_name = images.get(stem).name if stem in images else ""
            if kind == "vertebra_corner_pose":
                issues.append({"split": split, "label": label.name, "image": image_name, "kind": kind, "decision": "隔离", "reasons": "vertebra_corner_label_in_six_point_dataset"})
                continue
            if kind != "six_point_pose":
                reason = f"parse_error:{error}" if error else "invalid_or_unknown_pose_format"
                issues.append({"split": split, "label": label.name, "image": image_name, "kind": kind, "decision": "排除", "reasons": reason})
                continue
            values = rows[0]
            six_labels[(split, stem)] = values
            points = [tuple(values[5 + index * 3 : 8 + index * 3]) for index in range(6)]
            visibility.update(str(int(point[2])) for point in points)
            left_right_patterns.update(
                {
                    "CR_left_of_CL": int(points[0][0] < points[1][0]),
                    "IR_left_of_IL": int(points[2][0] < points[3][0]),
                    "SR_left_of_SL": int(points[4][0] < points[5][0]),
                }
            )
            reasons = six_point_reasons(values, threshold)
            if "all_visible_points_top_left" in reasons:
                split_six_point_quality[split]["top_left_systematic_error"] += 1
            elif reasons:
                split_six_point_quality[split]["needs_review"] += 1
            else:
                split_six_point_quality[split]["automatic_structure_pass"] += 1
            if reasons:
                decision = "隔离" if "all_visible_points_top_left" in reasons else "待复核"
                issues.append({"split": split, "label": label.name, "image": image_name, "kind": kind, "decision": decision, "reasons": "|".join(reasons)})

    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    duplicate_label_conflict_groups = 0
    for group in duplicate_groups:
        label_hashes: set[str] = set()
        for item in group:
            label = label_map(pose_dir, item["split"]).get(Path(item["image"]).stem)
            if label is not None:
                item["label_sha256"] = sha256(label)
                label_hashes.add(item["label_sha256"])
        if len(label_hashes) > 1:
            duplicate_label_conflict_groups += 1
    cross_split_duplicate_groups = [group for group in duplicate_groups if len({item["split"] for item in group}) > 1]

    detection_summary: dict[str, Any] | None = None
    if detection_dir is not None:
        valid = invalid = matched = mismatched = missing = extra = 0
        detection_labels: dict[tuple[str, str], list[list[float]]] = {}
        for split in SPLITS:
            for stem, label in sorted(label_map(detection_dir, split).items()):
                rows, error = read_rows(label)
                is_valid = error is None and bool(rows) and all(
                    len(row) == 5 and int(row[0]) in range(6) and all(0 <= value <= 1 for value in row[1:])
                    for row in rows
                )
                if is_valid:
                    valid += 1
                    detection_labels[(split, stem)] = rows
                else:
                    invalid += 1
        for key, values in six_labels.items():
            if key not in detection_labels:
                missing += 1
            elif rows_close(expected_detection(values), detection_labels[key]):
                matched += 1
            else:
                mismatched += 1
        extra = len(detection_labels.keys() - six_labels.keys())
        detection_summary = {
            "valid_labels": valid,
            "invalid_labels": invalid,
            "exact_pose_conversion_matches": matched,
            "conversion_mismatches": mismatched,
            "pose_labels_without_detection": missing,
            "detection_labels_without_pose": extra,
        }

    return {
        "schema_version": 1,
        "pose_dir": str(pose_dir.resolve()),
        "detection_dir": str(detection_dir.resolve()) if detection_dir else None,
        "top_left_threshold": threshold,
        "split_counts": split_counts,
        "split_label_types": {split: dict(counts) for split, counts in split_label_types.items()},
        "split_six_point_quality": {split: dict(counts) for split, counts in split_six_point_quality.items()},
        "pose_label_types": dict(sorted(pose_counts.items())),
        "six_point_visibility": dict(sorted(visibility.items())),
        "left_right_patterns": dict(sorted(left_right_patterns.items())),
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_label_conflict_groups": duplicate_label_conflict_groups,
        "cross_split_exact_duplicate_groups": cross_split_duplicate_groups,
        "detection": detection_summary,
        "issues": issues,
        "issue_decisions": dict(Counter(issue["decision"] for issue in issues)),
        "issue_reasons": dict(Counter(reason for issue in issues for reason in issue["reasons"].split("|"))),
    }


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "issues.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["issues"])
    counts = result["pose_label_types"]
    decisions = result["issue_decisions"]
    reasons = result["issue_reasons"]
    detection = result["detection"] or {}
    report = f"""# 旧六点数据集自动审计报告

## 核心结果

- Pose标签总数：{sum(counts.values())}
- 真六点Pose标签：{counts.get('six_point_pose', 0)}
- 误混入的椎体四角Pose标签：{counts.get('vertebra_corner_pose', 0)}
- 无法识别标签：{counts.get('invalid_or_unknown', 0)}
- 左上角系统性聚集：{reasons.get('all_visible_points_top_left', 0)}
- 锁骨左右顺序冲突：{reasons.get('clavicle_left_right_order_conflict', 0)}
- 髂骨左右顺序冲突：{reasons.get('iliac_left_right_order_conflict', 0)}
- 骶骨左右顺序冲突：{reasons.get('sacral_left_right_order_conflict', 0)}
- 骶骨点组位于髂骨点组上方：{reasons.get('sacral_pair_above_iliac_pair', 0)}
- bbox未包含所有可见点：{reasons.get('bbox_does_not_enclose_visible_points', 0)}
- 建议隔离：{decisions.get('隔离', 0)}
- 建议待复核：{decisions.get('待复核', 0)}
- 精确重复图像组：{len(result['exact_duplicate_groups'])}
- 精确重复但标签不一致组：{result['exact_duplicate_label_conflict_groups']}
- 跨split精确重复组：{len(result['cross_split_exact_duplicate_groups'])}

## Pose→Detection一致性

- Detection合法标签：{detection.get('valid_labels', 0)}
- 与现有转换逻辑精确一致：{detection.get('exact_pose_conversion_matches', 0)}
- 转换不一致：{detection.get('conversion_mismatches', 0)}

## 解释

`issues.csv` 是需隔离或复核的逐样本清单。Detection标签若与Pose转换结果一致，只说明派生过程一致，不是独立标注正确性证据。
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    result = audit(args.pose_dir, args.detection_dir, args.top_left_threshold)
    write_outputs(result, args.output)
    print(json.dumps({key: result[key] for key in ("pose_label_types", "issue_decisions", "issue_reasons", "detection")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
