#!/usr/bin/env python3
"""Restore historical Corner class 18 (L6) and class 19 (T13) labels incrementally.

The default mode is read-only.  ``--apply`` backs up every file that may change,
restores only the historical extra rows, refreshes affected Corner ROI views, and
rolls both layers back if any validation fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

import build_corner_roi_views as roi_builder


SPLITS = ("train", "val", "test")
BASE_CLASSES = frozenset(range(18))
EXTRA_CLASSES = frozenset({18, 19})
ALLOWED_CLASSES = BASE_CLASSES | EXTRA_CLASSES
IMAGE_SUFFIXES = roi_builder.IMAGE_SUFFIXES


@dataclass(frozen=True)
class LabelRow:
    class_id: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class RestoreItem:
    stem: str
    source_split: str
    active_split: str
    source_label: Path
    source_image: Path
    active_label: Path
    active_image: Path
    before_text: str
    after_text: str
    extra_classes: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_label_text(text: str, path: Path) -> tuple[LabelRow, ...]:
    rows: list[LabelRow] = []
    seen: set[int] = set()
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        tokens = raw.split()
        if len(tokens) != 17:
            raise ValueError(f"{path}:{line_number}: expected 17 fields, found {len(tokens)}")
        try:
            class_value = float(tokens[0])
            values = [float(token) for token in tokens[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric label field") from error
        if not class_value.is_integer() or int(class_value) not in ALLOWED_CLASSES:
            raise ValueError(f"{path}:{line_number}: invalid class id {tokens[0]}")
        class_id = int(class_value)
        if class_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate class id {class_id}")
        seen.add(class_id)
        coordinate_indices = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14)
        if any(not 0.0 <= values[index] <= 1.0 for index in coordinate_indices):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0,1]")
        if any(values[index] not in (0.0, 1.0, 2.0) for index in (6, 9, 12, 15)):
            raise ValueError(f"{path}:{line_number}: invalid visibility")
        rows.append(LabelRow(class_id, tuple(tokens)))
    if not rows:
        raise ValueError(f"{path}: empty label")
    return tuple(rows)


def parse_label(path: Path) -> tuple[LabelRow, ...]:
    return parse_label_text(path.read_text(encoding="utf-8"), path)


def canonical_extra_line(row: LabelRow) -> str:
    tokens = list(row.tokens)
    points = [(float(tokens[5 + index * 3]), float(tokens[6 + index * 3])) for index in range(4)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    bbox = (
        (left + right) / 2,
        (top + bottom) / 2,
        right - left,
        bottom - top,
    )
    return " ".join([str(row.class_id), *(f"{value:.8f}" for value in bbox), *tokens[5:]])


def keypoint_payload_matches(left: LabelRow, right: LabelRow, tolerance: float = 1e-8) -> bool:
    if left.class_id != right.class_id:
        return False
    left_values = [float(token) for token in left.tokens[5:]]
    right_values = [float(token) for token in right.tokens[5:]]
    return all(abs(a - b) <= tolerance for a, b in zip(left_values, right_values))


def index_unique_files(root: Path, pattern: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.glob(pattern)):
        if path.stem in result:
            raise ValueError(f"duplicate stem {path.stem}: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def find_image(image_root: Path, split: str, stem: str) -> Path:
    matches = [
        path
        for path in (image_root / split).glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one image for {split}/{stem}, found {len(matches)}")
    return matches[0]


def ensure_base_contract(rows: Iterable[LabelRow], path: Path) -> None:
    classes = {row.class_id for row in rows}
    missing = BASE_CLASSES - classes
    if missing not in (set(), {12}):
        raise ValueError(f"{path}: unexpected missing base classes {sorted(missing)}")


def build_restore_plan(
    source_label_root: Path,
    source_image_root: Path,
    corner_root: Path,
    *,
    expected_labels: int,
    expected_files: int,
    expected_class18: int,
    expected_class19: int,
) -> list[RestoreItem]:
    active_labels = index_unique_files(corner_root / "labels", "*/*.txt")
    if len(active_labels) != expected_labels:
        raise ValueError(f"active label count mismatch: actual={len(active_labels)} expected={expected_labels}")

    source_with_extras: list[tuple[str, Path, tuple[LabelRow, ...]]] = []
    source_stems: set[str] = set()
    extra_counts: Counter[int] = Counter()
    for split in SPLITS:
        for path in sorted((source_label_root / split).glob("*.txt")):
            rows = parse_label(path)
            extras = tuple(row for row in rows if row.class_id in EXTRA_CLASSES)
            if not extras:
                continue
            if path.stem in source_stems:
                raise ValueError(f"duplicate historical extra-label stem: {path.stem}")
            source_stems.add(path.stem)
            source_with_extras.append((split, path, rows))
            extra_counts.update(row.class_id for row in extras)

    if len(source_with_extras) != expected_files:
        raise ValueError(
            f"historical extra-label file count mismatch: actual={len(source_with_extras)} expected={expected_files}"
        )
    expected_counts = {18: expected_class18, 19: expected_class19}
    if {class_id: extra_counts[class_id] for class_id in sorted(EXTRA_CLASSES)} != expected_counts:
        raise ValueError(f"historical extra row count mismatch: actual={dict(extra_counts)} expected={expected_counts}")

    items: list[RestoreItem] = []
    for source_split, source_label, source_rows in source_with_extras:
        active_label = active_labels.get(source_label.stem)
        if active_label is None:
            raise FileNotFoundError(f"active label is missing for historical sample: {source_label.stem}")
        active_split = active_label.parent.name
        active_rows = parse_label(active_label)
        ensure_base_contract(source_rows, source_label)
        ensure_base_contract(active_rows, active_label)
        source_by_class = {row.class_id: row for row in source_rows}
        active_by_class = {row.class_id: row for row in active_rows}
        source_base = set(source_by_class) & BASE_CLASSES
        active_base = set(active_by_class) & BASE_CLASSES
        if source_base != active_base:
            raise ValueError(
                f"base class set mismatch for {source_label.stem}: source={sorted(source_base)} active={sorted(active_base)}"
            )
        for class_id in sorted(source_base):
            if not keypoint_payload_matches(source_by_class[class_id], active_by_class[class_id]):
                raise ValueError(f"base keypoint payload mismatch for {source_label.stem} class {class_id}")

        source_extras = {class_id: source_by_class[class_id] for class_id in EXTRA_CLASSES if class_id in source_by_class}
        active_extras = {class_id: active_by_class[class_id] for class_id in EXTRA_CLASSES if class_id in active_by_class}
        for class_id, row in active_extras.items():
            if class_id not in source_extras or not keypoint_payload_matches(source_extras[class_id], row):
                raise ValueError(f"active extra label conflicts with history for {source_label.stem} class {class_id}")

        output_lines = [" ".join(active_by_class[class_id].tokens) for class_id in sorted(active_base)]
        output_lines.extend(canonical_extra_line(source_extras[class_id]) for class_id in sorted(source_extras))
        after_text = "\n".join(output_lines) + "\n"
        parse_label_text(after_text, active_label)

        source_image = find_image(source_image_root, source_split, source_label.stem)
        active_image = find_image(corner_root / "images", active_split, source_label.stem)
        if sha256_file(source_image) != sha256_file(active_image):
            raise ValueError(f"source/active image hash mismatch for {source_label.stem}")
        items.append(
            RestoreItem(
                stem=source_label.stem,
                source_split=source_split,
                active_split=active_split,
                source_label=source_label,
                source_image=source_image,
                active_label=active_label,
                active_image=active_image,
                before_text=active_label.read_text(encoding="utf-8"),
                after_text=after_text,
                extra_classes=tuple(sorted(source_extras)),
            )
        )
    return sorted(items, key=lambda item: (item.active_split, item.stem))


def rows_as_corner_objects(text: str, path: Path) -> tuple[roi_builder.CornerObject, ...]:
    objects = []
    for row in parse_label_text(text, path):
        values = [float(token) for token in row.tokens]
        bbox = tuple(values[1:5])
        keypoints = tuple((values[offset], values[offset + 1], int(values[offset + 2])) for offset in range(5, 17, 3))
        objects.append(roi_builder.CornerObject(row.class_id, bbox, keypoints))
    return tuple(objects)


def roi_preflight(items: list[RestoreItem], corner_roi_root: Path) -> dict[str, Any]:
    manifest_path = corner_roi_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {Path(record["source_image"]).stem: record for record in manifest.get("records", [])}
    train_items = [item for item in items if item.active_split == "train"]
    contained = 0
    expansion_required: list[str] = []
    for item in train_items:
        record = records.get(item.stem)
        if record is None:
            raise ValueError(f"Corner ROI manifest is missing {item.stem}")
        with Image.open(item.active_image) as image:
            width, height = image.size
        box = roi_builder.CropBox(**record["crop_box"])
        if roi_builder.crop_contains_target(box, rows_as_corner_objects(item.after_text, item.active_label), width, height):
            contained += 1
        else:
            expansion_required.append(item.stem)
    return {
        "affected_train_files": len(train_items),
        "existing_crop_contains_restored_targets": contained,
        "existing_crop_requires_expansion": len(expansion_required),
        "expansion_required_stems": expansion_required,
    }


def summarize_plan(items: list[RestoreItem], roi_summary: dict[str, Any]) -> dict[str, Any]:
    split_files = Counter(item.active_split for item in items)
    class_rows = Counter(class_id for item in items for class_id in item.extra_classes)
    return {
        "schema_version": 1,
        "files_with_historical_extras": len(items),
        "changed_raw_labels": sum(item.before_text != item.after_text for item in items),
        "already_restored_raw_labels": sum(item.before_text == item.after_text for item in items),
        "restored_rows": sum(len(item.extra_classes) for item in items),
        "class_rows": {str(class_id): class_rows[class_id] for class_id in sorted(EXTRA_CLASSES)},
        "active_split_files": {split: split_files[split] for split in SPLITS},
        "roi": roi_summary,
    }


def ensure_empty_destination(path: Path, description: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{description} is not empty: {path}")


def backup_affected_files(
    items: list[RestoreItem],
    corner_root: Path,
    pose_roi_root: Path,
    corner_roi_root: Path,
    backup_root: Path,
) -> dict[str, Any]:
    ensure_empty_destination(backup_root, "backup root")
    backup_root.mkdir(parents=True, exist_ok=True)
    old_manifest_path = corner_roi_root / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_records = {Path(record["source_image"]).stem: record for record in old_manifest.get("records", [])}
    actions: list[dict[str, Any]] = []

    for item in items:
        raw_backup = backup_root / "raw_labels" / item.active_split / item.active_label.name
        atomic_copy(item.active_label, raw_backup)
        if sha256_file(raw_backup) != sha256_file(item.active_label):
            raise RuntimeError(f"raw label backup hash mismatch: {item.active_label}")
        action: dict[str, Any] = {
            "stem": item.stem,
            "source_split": item.source_split,
            "active_split": item.active_split,
            "extra_classes": list(item.extra_classes),
            "raw_label": str(item.active_label.relative_to(corner_root)),
            "raw_label_before_sha256": sha256_file(item.active_label),
            "raw_image": str(item.active_image.relative_to(corner_root)),
            "raw_image_sha256": sha256_file(item.active_image),
        }
        if item.active_split == "train":
            record = old_records.get(item.stem)
            if record is None:
                raise ValueError(f"Corner ROI manifest is missing {item.stem}")
            roi_label = corner_roi_root / record["output_label"]
            roi_image = corner_roi_root / record["output_image"]
            roi_label_backup = backup_root / "roi_labels" / roi_label.name
            atomic_copy(roi_label, roi_label_backup)
            action.update(
                roi_output_mode_before=record["output_mode"],
                roi_reuse_image=record.get("reuse_image"),
                roi_label=str(roi_label.relative_to(corner_roi_root)),
                roi_label_before_sha256=sha256_file(roi_label),
                roi_image=str(roi_image.relative_to(corner_roi_root)),
                roi_image_before_sha256=sha256_file(roi_image),
            )
            if record["output_mode"] == "generated":
                roi_image_backup = backup_root / "roi_images" / roi_image.name
                atomic_copy(roi_image, roi_image_backup)
                if sha256_file(roi_image_backup) != action["roi_image_before_sha256"]:
                    raise RuntimeError(f"ROI image backup hash mismatch: {roi_image}")
            elif record["output_mode"] != "hardlink" or not record.get("reuse_image"):
                raise ValueError(f"unsupported previous ROI output mode for {item.stem}: {record['output_mode']}")
        actions.append(action)

    metadata = {}
    for name in ("manifest.json", "README.md"):
        source = corner_roi_root / name
        destination = backup_root / "roi_metadata" / name
        atomic_copy(source, destination)
        metadata[name] = sha256_file(source)
        if sha256_file(destination) != metadata[name]:
            raise RuntimeError(f"ROI metadata backup hash mismatch: {source}")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "ready",
        "corner_root": str(corner_root.resolve()),
        "pose_roi_root": str(pose_roi_root.resolve()),
        "corner_roi_root": str(corner_roi_root.resolve()),
        "roi_metadata_before_sha256": metadata,
        "actions": actions,
    }
    atomic_write_text(backup_root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def restore_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.rollback")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def rollback_from_backup(
    backup_manifest: dict[str, Any],
    backup_root: Path,
    corner_root: Path,
    pose_roi_root: Path,
    corner_roi_root: Path,
) -> None:
    for action in backup_manifest["actions"]:
        raw_label = corner_root / action["raw_label"]
        atomic_copy(backup_root / "raw_labels" / action["active_split"] / raw_label.name, raw_label)
        if action["active_split"] != "train":
            continue
        roi_label = corner_roi_root / action["roi_label"]
        roi_image = corner_roi_root / action["roi_image"]
        atomic_copy(backup_root / "roi_labels" / roi_label.name, roi_label)
        if action["roi_output_mode_before"] == "hardlink":
            restore_hardlink(pose_roi_root / action["roi_reuse_image"], roi_image)
        else:
            atomic_copy(backup_root / "roi_images" / roi_image.name, roi_image)
    for name in ("manifest.json", "README.md"):
        atomic_copy(backup_root / "roi_metadata" / name, corner_roi_root / name)


def count_extra_rows(label_paths: Iterable[Path]) -> Counter[int]:
    result: Counter[int] = Counter()
    for path in label_paths:
        result.update(row.class_id for row in parse_label(path) if row.class_id in EXTRA_CLASSES)
    return result


def apply_restoration(
    items: list[RestoreItem],
    plan_summary: dict[str, Any],
    corner_root: Path,
    pose_roi_root: Path,
    corner_roi_root: Path,
    backup_root: Path,
    record_dir: Path,
    *,
    expected_labels: int,
    expected_roi_labels: int,
    expected_class18: int,
    expected_class19: int,
) -> dict[str, Any]:
    changed = [item for item in items if item.before_text != item.after_text]
    if not changed:
        return {**plan_summary, "mode": "apply", "status": "already_restored"}
    ensure_empty_destination(record_dir, "record directory")
    backup_manifest = backup_affected_files(items, corner_root, pose_roi_root, corner_roi_root, backup_root)
    try:
        for item in changed:
            atomic_write_text(item.active_label, item.after_text)
        all_label_paths = sorted((corner_root / "labels").glob("*/*.txt"))
        if len(all_label_paths) != expected_labels:
            raise RuntimeError("raw label count changed during restoration")
        expected_extra = Counter({18: expected_class18, 19: expected_class19})
        if count_extra_rows(all_label_paths) != expected_extra:
            raise RuntimeError("raw restored extra-class counts do not match safety gate")

        records, configuration = roi_builder.plan_views(corner_root, pose_roi_root)
        if len(records) != expected_roi_labels:
            raise RuntimeError(f"ROI source count mismatch: actual={len(records)} expected={expected_roi_labels}")
        roi_manifest = roi_builder.apply_plan(records, configuration, corner_root, pose_roi_root, corner_roi_root)
        affected_train = sum(item.active_split == "train" for item in items)
        expected_skipped = expected_roi_labels - affected_train
        if roi_manifest["summary"]["skipped_existing_count"] != expected_skipped:
            raise RuntimeError(
                "incremental ROI update touched an unexpected number of files: "
                f"skipped={roi_manifest['summary']['skipped_existing_count']} expected={expected_skipped}"
            )
        roi_label_paths = sorted((corner_roi_root / "labels" / "train").glob("*.txt"))
        expected_train_extra = Counter(
            class_id for item in items if item.active_split == "train" for class_id in item.extra_classes
        )
        if count_extra_rows(roi_label_paths) != expected_train_extra:
            raise RuntimeError("ROI extra-class counts do not match restored train labels")

        actions_by_stem = {action["stem"]: action for action in backup_manifest["actions"]}
        new_roi_records = {Path(record["source_image"]).stem: record for record in roi_manifest["records"]}
        for item in items:
            action = actions_by_stem[item.stem]
            action["raw_label_after_sha256"] = sha256_file(item.active_label)
            if sha256_file(item.active_image) != action["raw_image_sha256"]:
                raise RuntimeError(f"raw image changed unexpectedly: {item.active_image}")
            if item.active_split == "train":
                record = new_roi_records[item.stem]
                action.update(
                    roi_output_mode_after=record["output_mode"],
                    roi_plan_reason_after=record["plan_reason"],
                    roi_crop_box_after=record["crop_box"],
                    roi_label_after_sha256=record["output_label_sha256"],
                    roi_image_after_sha256=record["output_image_sha256"],
                )

        result = {
            **plan_summary,
            "mode": "apply",
            "status": "restored",
            "completed_at": datetime.now().astimezone().isoformat(),
            "backup_root": str(backup_root.resolve()),
            "record_dir": str(record_dir.resolve()),
            "roi_update_summary": roi_manifest["summary"],
            "actions": backup_manifest["actions"],
        }
        backup_manifest.update(status="applied", completed_at=result["completed_at"], result=result)
        atomic_write_text(backup_root / "manifest.json", json.dumps(backup_manifest, ensure_ascii=False, indent=2) + "\n")
        record_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(record_dir / "manifest.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        atomic_write_text(
            record_dir / "README.md",
            "# Corner V18/V19增量恢复记录\n\n"
            "仅恢复历史标签中的class 18（L6）和class 19（T13）；原有class 0–17、图像和split未改变。\n\n"
            "受影响的train样本已同步刷新Corner ROI标签/裁剪，完整逐文件哈希见manifest.json。\n",
        )
        return result
    except Exception as error:
        rollback_from_backup(backup_manifest, backup_root, corner_root, pose_roi_root, corner_roi_root)
        backup_manifest.update(
            status="rolled_back",
            failed_at=datetime.now().astimezone().isoformat(),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_write_text(backup_root / "manifest.json", json.dumps(backup_manifest, ensure_ascii=False, indent=2) + "\n")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-label-root", type=Path, default=Path("/Users/liruirui/Downloads/labels"))
    parser.add_argument("--source-image-root", type=Path, default=Path("/Users/liruirui/Downloads/images"))
    parser.add_argument("--corner-root", type=Path, default=Path("datasets/pose_corner_data"))
    parser.add_argument("--pose-roi-root", type=Path, default=Path("datasets/pose_roi_views"))
    parser.add_argument("--corner-roi-root", type=Path, default=Path("datasets/corner_roi_views"))
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--expected-labels", type=int, default=2499)
    parser.add_argument("--expected-roi-labels", type=int, default=1999)
    parser.add_argument("--expected-files", type=int, default=54)
    parser.add_argument("--expected-class18", type=int, default=44)
    parser.add_argument("--expected-class19", type=int, default=11)
    parser.add_argument("--apply", action="store_true", help="back up and apply; default is read-only dry-run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = build_restore_plan(
        args.source_label_root,
        args.source_image_root,
        args.corner_root,
        expected_labels=args.expected_labels,
        expected_files=args.expected_files,
        expected_class18=args.expected_class18,
        expected_class19=args.expected_class19,
    )
    roi_summary = roi_preflight(items, args.corner_roi_root)
    plan_summary = summarize_plan(items, roi_summary)
    if args.apply:
        if args.backup_root is None or args.record_dir is None:
            raise ValueError("--apply requires --backup-root and --record-dir")
        result = apply_restoration(
            items,
            plan_summary,
            args.corner_root,
            args.pose_roi_root,
            args.corner_roi_root,
            args.backup_root,
            args.record_dir,
            expected_labels=args.expected_labels,
            expected_roi_labels=args.expected_roi_labels,
            expected_class18=args.expected_class18,
            expected_class19=args.expected_class19,
        )
    else:
        result = {**plan_summary, "mode": "dry_run", "status": "planned", "output_written": False}
    print(json.dumps({key: value for key, value in result.items() if key != "actions"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
