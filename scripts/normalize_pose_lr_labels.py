#!/usr/bin/env python3
"""Normalize six-point YOLO Pose labels to screen-left L / screen-right R."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PAIR_STARTS = ((5, 8), (11, 14), (17, 20))


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


def parse_label(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected one non-empty row, got {len(lines)}")
    tokens = lines[0].split()
    if len(tokens) != 23 or tokens[0] != "0":
        raise ValueError(f"{path}: expected class 0 and 23 tokens")
    values = [float(token) for token in tokens]
    if any(not 0.0 <= values[index] <= 1.0 for index in range(1, 23) if index not in (7, 10, 13, 16, 19, 22)):
        raise ValueError(f"{path}: bbox or keypoint coordinate outside [0, 1]")
    return tokens


def lr_pattern(tokens: list[str]) -> str:
    # Slots are CR,CL / IR,IL / SR,SL. New domain convention has L on screen left.
    new_relations = [float(tokens[left]) < float(tokens[right]) for right, left in PAIR_STARTS]
    if all(new_relations):
        return "normalized_L_on_screen_left"
    if not any(new_relations):
        return "legacy_R_on_screen_left"
    return "mixed"


def swap_lr_pairs(tokens: list[str]) -> list[str]:
    swapped = list(tokens)
    for first, second in PAIR_STARTS:
        swapped[first : first + 3], swapped[second : second + 3] = (
            tokens[second : second + 3],
            tokens[first : first + 3],
        )
    return swapped


def expected_detection(tokens: list[str], bbox_ratio: float = 0.04) -> str:
    half = bbox_ratio / 2
    lines: list[str] = []
    for keypoint_index in range(6):
        start = 5 + keypoint_index * 3
        x, y, visibility = map(float, tokens[start : start + 3])
        if int(visibility) == 0:
            continue
        cx = max(half, min(1 - half, x))
        cy = max(half, min(1 - half, y))
        lines.append(
            f"{keypoint_index} {cx:.6f} {cy:.6f} {bbox_ratio:.6f} {bbox_ratio:.6f}"
        )
    return "\n".join(lines) + "\n"


def sync_derived_detection(
    dataset: Path,
    detection: Path,
    backup: Path | None = None,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    detection = detection.resolve()
    matched: list[tuple[Path, Path, str]] = []
    orphan_paths: list[Path] = []
    for detection_label in sorted((detection / "labels").glob("*/*.txt")):
        relative = detection_label.relative_to(detection / "labels")
        pose_label = dataset / "labels" / relative
        if pose_label.is_file():
            matched.append((pose_label, detection_label, expected_detection(parse_label(pose_label))))
        else:
            orphan_paths.append(detection_label)
    changed = [item for item in matched if item[1].read_text(encoding="utf-8") != item[2]]
    orphan_digest_before = tree_digest(detection, orphan_paths)
    result: dict[str, Any] = {
        "schema_version": 1,
        "pose_dataset": str(dataset),
        "detection_dataset": str(detection),
        "mode": "apply" if apply else "dry_run",
        "matched_labels": len(matched),
        "changed_labels": len(changed),
        "orphan_labels_untouched": len(orphan_paths),
        "orphan_tree_sha256_before": orphan_digest_before,
    }
    if not apply:
        return result
    if not changed:
        result.update(status="already_synchronized")
        return result
    if backup is None:
        raise ValueError("backup directory is required when synchronizing with --apply")
    backup = backup.resolve()
    actions: list[dict[str, str]] = []
    for _, detection_label, expected in changed:
        relative = detection_label.relative_to(detection / "labels")
        destination = backup / "detection_labels" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before_sha256 = sha256_file(detection_label)
        shutil.copy2(detection_label, destination)
        if sha256_file(destination) != before_sha256:
            raise RuntimeError(f"detection backup verification failed: {detection_label}")
        temporary = detection_label.with_name(f".{detection_label.name}.sync.tmp")
        temporary.write_text(expected, encoding="utf-8")
        temporary.replace(detection_label)
        actions.append(
            {
                "label": str(relative),
                "before_sha256": before_sha256,
                "after_sha256": sha256_file(detection_label),
            }
        )
    for pose_label, detection_label, expected in matched:
        if detection_label.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"derived detection verification failed: {pose_label}")
    orphan_digest_after = tree_digest(detection, orphan_paths)
    if orphan_digest_before != orphan_digest_after:
        raise RuntimeError("orphan detection labels changed unexpectedly")
    result.update(
        status="synchronized",
        orphan_tree_sha256_after=orphan_digest_after,
        actions=actions,
    )
    (backup / "detection_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def normalize_dataset(dataset: Path, backup: Path | None = None, *, apply: bool = False) -> dict[str, Any]:
    dataset = dataset.resolve()
    label_paths = sorted((dataset / "labels").glob("*/*.txt"))
    image_paths = sorted(path for path in (dataset / "images").glob("*/*") if path.is_file() and not path.name.startswith("."))
    if not label_paths:
        raise ValueError(f"no labels found under {dataset / 'labels'}")

    parsed = {path: parse_label(path) for path in label_paths}
    patterns = Counter(lr_pattern(tokens) for tokens in parsed.values())
    if patterns.get("mixed"):
        raise ValueError(f"mixed left/right labels must be reviewed manually: {patterns['mixed']}")
    if patterns.get("legacy_R_on_screen_left") and patterns.get("normalized_L_on_screen_left"):
        raise ValueError(f"dataset contains both legacy and normalized conventions: {dict(patterns)}")

    before_label_digest = tree_digest(dataset, label_paths)
    before_image_digest = tree_digest(dataset, image_paths)
    actions: list[dict[str, str]] = []
    for path, tokens in parsed.items():
        if lr_pattern(tokens) != "legacy_R_on_screen_left":
            continue
        swapped = swap_lr_pairs(tokens)
        if lr_pattern(swapped) != "normalized_L_on_screen_left":
            raise AssertionError(f"swap did not normalize {path}")
        actions.append(
            {
                "label": str(path.relative_to(dataset)),
                "before_sha256": sha256_file(path),
                "after_text": " ".join(swapped) + "\n",
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(dataset),
        "mode": "apply" if apply else "dry_run",
        "label_count": len(label_paths),
        "image_count": len(image_paths),
        "patterns_before": dict(sorted(patterns.items())),
        "changed_labels": len(actions),
        "label_tree_sha256_before": before_label_digest,
        "image_tree_sha256_before": before_image_digest,
    }
    if not apply:
        return result
    if not actions:
        result.update(status="already_normalized", patterns_after=dict(sorted(patterns.items())))
        return result
    if backup is None:
        raise ValueError("backup directory is required with --apply")
    backup = backup.resolve()
    if backup.exists():
        raise FileExistsError(f"backup directory already exists: {backup}")

    originals = backup / "labels"
    for action in actions:
        source = dataset / action["label"]
        destination = originals / Path(action["label"]).relative_to("labels")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != action["before_sha256"]:
            raise RuntimeError(f"backup verification failed: {source}")

    for action in actions:
        path = dataset / action["label"]
        temporary = path.with_name(f".{path.name}.lr-normalize.tmp")
        temporary.write_text(action.pop("after_text"), encoding="utf-8")
        temporary.replace(path)
        action["after_sha256"] = sha256_file(path)

    after_parsed = {path: parse_label(path) for path in label_paths}
    after_patterns = Counter(lr_pattern(tokens) for tokens in after_parsed.values())
    after_image_digest = tree_digest(dataset, image_paths)
    if after_patterns != {"normalized_L_on_screen_left": len(label_paths)}:
        raise RuntimeError(f"post-normalization pattern verification failed: {dict(after_patterns)}")
    if before_image_digest != after_image_digest:
        raise RuntimeError("image tree changed while normalizing labels")

    result.update(
        {
            "status": "normalized",
            "backup": str(backup),
            "patterns_after": dict(sorted(after_patterns.items())),
            "label_tree_sha256_after": tree_digest(dataset, label_paths),
            "image_tree_sha256_after": after_image_digest,
            "actions": actions,
        }
    )
    backup.mkdir(parents=True, exist_ok=True)
    (backup / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (backup / "README.md").write_text(
        "# 旧六点标签左右语义修正备份\n\n"
        "本目录保存修正前的原始YOLO Pose标签。活动数据只交换了"
        "CR/CL、IR/IL、SR/SL三对关键点槽位；图像、bbox、坐标数值和split均未改变。\n\n"
        "恢复时可按原相对路径将 `labels/` 内文件复制回数据集。逐文件SHA-256见 `manifest.json`。\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--detection-dir", type=Path, help="同步现有Pose派生Detection标签")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = normalize_dataset(args.dataset, args.backup, apply=args.apply)
    output: dict[str, Any] = {"pose": result}
    if args.detection_dir is not None:
        output["detection"] = sync_derived_detection(
            args.dataset, args.detection_dir, args.backup, apply=args.apply
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
