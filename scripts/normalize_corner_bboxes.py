#!/usr/bin/env python3
"""Normalize YOLO Pose corner bboxes to the axis-aligned envelope of four keypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
BBOX_TOLERANCE = 2e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_bbox(tokens: list[str]) -> list[str]:
    """Return cx/cy/w/h from TL/TR/BR/BL while retaining eight decimal places."""
    points = [(float(tokens[5 + 3 * index]), float(tokens[6 + 3 * index])) for index in range(4)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return [
        f"{(left + right) / 2:.8f}",
        f"{(top + bottom) / 2:.8f}",
        f"{right - left:.8f}",
        f"{bottom - top:.8f}",
    ]


def bbox_is_canonical(tokens: list[str]) -> bool:
    expected = [float(value) for value in canonical_bbox(tokens)]
    actual = [float(value) for value in tokens[1:5]]
    return max(abs(old - new) for old, new in zip(actual, expected)) <= BBOX_TOLERANCE


def normalize_label_text(path: Path) -> tuple[str, int, int]:
    output: list[str] = []
    changed_rows = 0
    total_rows = 0
    classes: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        tokens = raw.split()
        if len(tokens) != 17:
            raise ValueError(f"标签列数不是17：{path}:{line_number}")
        try:
            class_id = int(tokens[0])
            values = [float(token) for token in tokens[1:]]
        except ValueError as error:
            raise ValueError(f"标签含非数值字段：{path}:{line_number}") from error
        if not 0 <= class_id <= 17:
            raise ValueError(f"类别越界：{path}:{line_number} class={class_id}")
        coordinate_indices = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14)
        if any(not 0.0 <= values[index] <= 1.0 for index in coordinate_indices):
            raise ValueError(f"坐标越界：{path}:{line_number}")
        if any(values[index] not in (0.0, 1.0, 2.0) for index in (6, 9, 12, 15)):
            raise ValueError(f"visibility非法：{path}:{line_number}")
        bbox = tokens[1:5] if bbox_is_canonical(tokens) else canonical_bbox(tokens)
        normalized = [tokens[0], *bbox, *tokens[5:]]
        if tokens[1:5] != bbox:
            changed_rows += 1
        output.append(" ".join(normalized))
        total_rows += 1
        classes.append(class_id)
    if not output:
        raise ValueError(f"空标签：{path}")
    if len(classes) != len(set(classes)):
        raise ValueError(f"存在重复椎体类别：{path}")
    missing = sorted(set(range(18)) - set(classes))
    if missing not in ([], [12]):
        raise ValueError(f"缺失类别超出已知例外：{path} missing={missing}")
    return "\n".join(output) + "\n", changed_rows, total_rows


def verify_keypoint_fields(before: str, after: str, path: Path) -> None:
    before_rows = [line.split() for line in before.splitlines() if line.strip()]
    after_rows = [line.split() for line in after.splitlines() if line.strip()]
    if len(before_rows) != len(after_rows):
        raise RuntimeError(f"行数发生变化：{path}")
    for line_number, (old, new) in enumerate(zip(before_rows, after_rows), 1):
        if old[0] != new[0] or old[5:] != new[5:]:
            raise RuntimeError(f"bbox之外字段发生变化：{path}:{line_number}")


def audit_normalized(label_paths: list[Path]) -> dict[str, Any]:
    issues: list[str] = []
    rows = 0
    split_files = Counter()
    class_counts = Counter()
    for path in label_paths:
        split_files[path.parent.name] += 1
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            tokens = raw.split()
            rows += 1
            if len(tokens) != 17:
                issues.append(f"{path}:{line_number}: fields={len(tokens)}")
                continue
            class_counts[int(tokens[0])] += 1
            if not bbox_is_canonical(tokens):
                issues.append(f"{path}:{line_number}: bbox_not_canonical")
    return {
        "labels": len(label_paths),
        "rows": rows,
        "split_labels": {split: split_files[split] for split in SPLITS},
        "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "issues": issues,
    }


def normalize_dataset(
    dataset: Path,
    backup: Path | None = None,
    *,
    apply: bool = False,
    expected_labels: int | None = None,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    label_root = dataset / "labels"
    label_paths = sorted(label_root.glob("*/*.txt"))
    image_paths = sorted(path for path in (dataset / "images").glob("*/*") if path.is_file())
    if expected_labels is not None and len(label_paths) != expected_labels:
        raise ValueError(f"标签规模与安全门槛不符：actual={len(label_paths)} expected={expected_labels}")
    if len(image_paths) != len(label_paths):
        raise ValueError(f"图像标签数量不一致：images={len(image_paths)} labels={len(label_paths)}")

    planned: dict[Path, tuple[str, str, int, int]] = {}
    changed_files = 0
    changed_rows = 0
    total_rows = 0
    for path in label_paths:
        before = path.read_text(encoding="utf-8")
        after, file_changed_rows, file_rows = normalize_label_text(path)
        verify_keypoint_fields(before, after, path)
        planned[path] = (before, after, file_changed_rows, file_rows)
        changed_files += before != after
        changed_rows += file_changed_rows
        total_rows += file_rows

    result: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(dataset),
        "rule": "axis_aligned_minimum_envelope_of_TL_TR_BR_BL_no_padding",
        "mode": "apply" if apply else "dry_run",
        "labels": len(label_paths),
        "rows": total_rows,
        "changed_files": changed_files,
        "changed_rows": changed_rows,
        "label_tree_sha256_before": tree_digest(dataset, label_paths),
        "image_tree_sha256_before": tree_digest(dataset, image_paths),
    }
    if not apply:
        return result
    if changed_files == 0:
        result.update(status="already_normalized", audit=audit_normalized(label_paths))
        return result
    if backup is None:
        raise ValueError("正式修改必须提供backup目录")
    backup = backup.resolve()
    if backup.exists() and any(backup.iterdir()):
        raise FileExistsError(f"备份目录非空：{backup}")

    actions: list[dict[str, Any]] = []
    written: list[Path] = []
    try:
        for path in label_paths:
            before, _, file_changed_rows, file_rows = planned[path]
            relative = path.relative_to(label_root)
            destination = backup / "labels" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            before_sha256 = sha256_file(path)
            if sha256_file(destination) != before_sha256:
                raise RuntimeError(f"备份哈希不一致：{path}")
            actions.append(
                {
                    "label": str(relative),
                    "rows": file_rows,
                    "changed_rows": file_changed_rows,
                    "before_sha256": before_sha256,
                }
            )

        for path in label_paths:
            before, after, _, _ = planned[path]
            if before == after:
                continue
            temporary = path.with_name(f".{path.name}.bbox.tmp")
            temporary.write_text(after, encoding="utf-8")
            temporary.replace(path)
            written.append(path)

        for action, path in zip(actions, label_paths):
            action["after_sha256"] = sha256_file(path)
            backup_path = backup / "labels" / Path(action["label"])
            if sha256_file(backup_path) != action["before_sha256"]:
                raise RuntimeError(f"备份复验失败：{path}")
            verify_keypoint_fields(backup_path.read_text(encoding="utf-8"), path.read_text(encoding="utf-8"), path)

        audit = audit_normalized(label_paths)
        if audit["issues"]:
            raise RuntimeError(f"统一后审计失败：{audit['issues'][:5]}")
        result.update(
            status="normalized",
            label_tree_sha256_after=tree_digest(dataset, label_paths),
            image_tree_sha256_after=tree_digest(dataset, image_paths),
            audit=audit,
            actions=actions,
        )
        if result["image_tree_sha256_before"] != result["image_tree_sha256_after"]:
            raise RuntimeError("图像目录发生变化")
        backup.mkdir(parents=True, exist_ok=True)
        (backup / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (backup / "README.md").write_text(
            "# pose_corner_data bbox统一前标签备份\n\n"
            f"执行时间：{datetime.now().astimezone().isoformat()}\n\n"
            "活动标签的bbox已统一为TL/TR/BR/BL四点的水平最小外接框，不加边距。\n"
            "本目录保留修改前全部标签；manifest.json记录逐文件修改前后SHA-256。\n"
            "图像、类别、角点、visibility、行序和split均未修改。\n",
            encoding="utf-8",
        )
        return result
    except Exception:
        for path in written:
            relative = path.relative_to(label_root)
            source = backup / "labels" / relative
            if source.is_file():
                shutil.copy2(source, path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一椎体四角Pose标签的bbox定义")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--expected-labels", type=int)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = normalize_dataset(
        args.dataset,
        args.backup,
        apply=args.apply,
        expected_labels=args.expected_labels,
    )
    summary = {key: value for key, value in result.items() if key != "actions"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
