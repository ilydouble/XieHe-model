#!/usr/bin/env python3
"""Apply approved duplicate-review selections to the two YOLO Pose datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from build_task_training_manifests import read_annotation
from import_e_drive_training_data import corner_yolo, six_point_yolo


TASK_ROOT_KEYS = {"six_point": "pose_root", "spine_pose": "corner_root"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_text(task: str, annotation: Path) -> str:
    data = read_annotation(annotation)
    return six_point_yolo(data, lr_policy="preserve") if task == "six_point" else corner_yolo(data)


def find_image_by_hash(root: Path, digest: str) -> Path:
    matches = [
        path
        for split in ("train", "val", "test")
        for path in (root / "images" / split).glob("*")
        if path.is_file() and not path.name.startswith(".") and sha256_file(path) == digest
    ]
    if len(matches) != 1:
        raise ValueError(f"期望SHA-256在目标数据中唯一，实际{len(matches)}：{digest}")
    return matches[0]


def build_plan(analysis_path: Path, pose_root: Path, corner_root: Path) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    roots = {"six_point": pose_root, "spine_pose": corner_root}
    additions: list[dict[str, Any]] = []
    for record in analysis["records"]:
        if record["status"] != "eligible_new":
            continue
        task = record["task"]
        image = Path(record["source_image"])
        annotation = Path(record["source_annotation"])
        text = label_text(task, annotation)
        destination_image = roots[task] / "images/train" / f"eap_{image.name}"
        destination_label = roots[task] / "labels/train" / f"eap_{image.stem}.txt"
        if destination_image.exists() or destination_label.exists():
            raise FileExistsError(f"新增目标已存在：{destination_image}")
        additions.append({
            "group": record["group"], "task": task,
            "source_image": str(image.resolve()), "source_annotation": str(annotation.resolve()),
            "source_image_sha256": sha256_file(image),
            "generated_label_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "destination_image": str(destination_image.resolve()),
            "destination_label": str(destination_label.resolve()),
        })

    removals = []
    for record in analysis["records"]:
        if record["task"] != "six_point" or record["status"] != "present_but_rejected":
            continue
        image = find_image_by_hash(pose_root, record["sha256"])
        label = pose_root / "labels" / image.parent.name / f"{image.stem}.txt"
        removals.append({
            "group": record["group"], "task": "six_point", "image": str(image.resolve()),
            "label": str(label.resolve()), "image_sha256": sha256_file(image), "label_sha256": sha256_file(label),
        })

    replacements = []
    for record in analysis["records"]:
        if record["task"] != "spine_pose" or record["status"] != "needs_label_replacement":
            continue
        image = find_image_by_hash(corner_root, record["sha256"])
        current_label = corner_root / "labels" / image.parent.name / f"{image.stem}.txt"
        annotation = Path(record["source_annotation"])
        text = label_text("spine_pose", annotation)
        replacements.append({
            "group": record["group"], "task": "spine_pose", "image": str(image.resolve()),
            "label": str(current_label.resolve()), "old_label_sha256": sha256_file(current_label),
            "source_annotation": str(annotation.resolve()),
            "new_label_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })

    counts = Counter(item["task"] for item in additions)
    if (counts["six_point"], counts["spine_pose"], len(removals), len(replacements)) != (263, 247, 2, 1):
        raise ValueError(f"计划规模未达到确认门槛：{counts}, removals={len(removals)}, replacements={len(replacements)}")
    return {
        "schema_version": 1, "generated_at": datetime.now().astimezone().isoformat(),
        "analysis": str(analysis_path.resolve()),
        "pose_root": str(pose_root.resolve()), "corner_root": str(corner_root.resolve()),
        "expected_before": {"six_point": 1494, "spine_pose": 2252},
        "expected_after": {"six_point": 1755, "spine_pose": 2499},
        "additions": additions, "removals": removals, "replacements": replacements,
    }


def count_pairs(root: Path) -> tuple[int, dict[str, int]]:
    counts = {}
    for split in ("train", "val", "test"):
        images = {path.stem for path in (root / "images" / split).glob("*") if path.is_file() and not path.name.startswith(".")}
        labels = {path.stem for path in (root / "labels" / split).glob("*.txt") if path.is_file() and not path.name.startswith(".")}
        if images != labels:
            raise ValueError(f"{root}/{split}图像标签不配对")
        counts[split] = len(images)
    return sum(counts.values()), counts


def apply_plan(plan: dict[str, Any], quarantine: Path) -> None:
    roots = {task: Path(plan[key]) for task, key in TASK_ROOT_KEYS.items()}
    for task, expected in plan["expected_before"].items():
        if count_pairs(roots[task])[0] != expected:
            raise ValueError(f"{task}应用前数量不符合门槛")
    if quarantine.exists() and any(quarantine.iterdir()):
        raise FileExistsError(f"备份目录非空：{quarantine}")
    created: list[Path] = []
    moved: list[tuple[Path, Path]] = []
    replacement_backups: list[tuple[Path, Path]] = []
    try:
        for record in plan["additions"]:
            image = Path(record["source_image"]); annotation = Path(record["source_annotation"])
            destination_image = Path(record["destination_image"]); destination_label = Path(record["destination_label"])
            destination_image.parent.mkdir(parents=True, exist_ok=True); destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination_image); created.append(destination_image)
            destination_label.write_text(label_text(record["task"], annotation), encoding="utf-8"); created.append(destination_label)
            if sha256_file(destination_image) != record["source_image_sha256"] or sha256_file(destination_label) != record["generated_label_sha256"]:
                raise RuntimeError(f"新增文件哈希验证失败：{record['group']}")
        for record in plan["removals"]:
            for kind in ("image", "label"):
                source = Path(record[kind]); destination = quarantine / "removed" / record["group"] / source.name
                destination.parent.mkdir(parents=True, exist_ok=True); shutil.move(source, destination); moved.append((source, destination))
                if sha256_file(destination) != record[f"{kind}_sha256"]:
                    raise RuntimeError(f"隔离文件哈希验证失败：{source}")
                record[f"quarantine_{kind}"] = str(destination.resolve())
        for record in plan["replacements"]:
            label = Path(record["label"]); backup = quarantine / "replaced" / record["group"] / label.name
            backup.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(label, backup); replacement_backups.append((label, backup))
            if sha256_file(backup) != record["old_label_sha256"]:
                raise RuntimeError(f"替换备份哈希验证失败：{label}")
            label.write_text(label_text("spine_pose", Path(record["source_annotation"])), encoding="utf-8")
            if sha256_file(label) != record["new_label_sha256"]:
                raise RuntimeError(f"替换标签哈希验证失败：{label}")
            record["backup_label"] = str(backup.resolve())
    except Exception:
        for label, backup in reversed(replacement_backups): shutil.copy2(backup, label)
        for source, destination in reversed(moved): source.parent.mkdir(parents=True, exist_ok=True); shutil.move(destination, source)
        for path in reversed(created): path.unlink(missing_ok=True)
        raise


def audit_dataset(root: Path, task: str) -> dict[str, Any]:
    total, splits = count_pairs(root)
    hashes: dict[str, list[str]] = {}; issues = []
    for split in ("train", "val", "test"):
        for image in (root / "images" / split).glob("*"):
            if not image.is_file() or image.name.startswith("."): continue
            try:
                with Image.open(image) as source: source.load()
            except Exception as error: issues.append(f"图像解码失败:{image}:{error}")
            hashes.setdefault(sha256_file(image), []).append(f"{split}/{image.name}")
            label = root / "labels" / split / f"{image.stem}.txt"
            try:
                lines = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
                expected_columns = 23 if task == "six_point" else 17
                if task == "six_point" and (len(lines) != 1 or len(lines[0]) != expected_columns): raise ValueError("六点标签结构异常")
                if task == "spine_pose" and (not lines or any(len(line) != expected_columns for line in lines)): raise ValueError("角点标签结构异常")
                classes = []
                for line in lines:
                    class_id = int(line[0]); classes.append(class_id)
                    values = [float(value) for value in line[1:]]
                    coordinate_indices = (
                        (0,1,2,3,4,5,7,8,10,11,13,14,16,17,19,20)
                        if task == "six_point"
                        else (0,1,2,3,4,5,7,8,10,11,13,14)
                    )
                    if any(not 0 <= values[i] <= 1 for i in coordinate_indices): raise ValueError("坐标越界")
                    visibility_indices = (6,9,12,15,18,21) if task == "six_point" else (6,9,12,15)
                    if any(values[i] not in (0,1,2) for i in visibility_indices): raise ValueError("visibility非法")
                    if task == "six_point":
                        if class_id != 0: raise ValueError("六点类别不是0")
                        xs = [values[i] for i in (4,7,10,13,16,19)]
                        if not (xs[0] > xs[1] and xs[2] > xs[3] and xs[4] > xs[5]): raise ValueError("六点左右顺序异常")
                    else:
                        if not 0 <= class_id <= 17: raise ValueError("角点类别越界")
                        tl, tr, br, bl = ((values[4],values[5]),(values[7],values[8]),(values[10],values[11]),(values[13],values[14]))
                        if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] + tr[1] < bl[1] + br[1]): raise ValueError("角点几何顺序异常")
                if task == "spine_pose" and len(classes) != len(set(classes)): raise ValueError("角点类别重复")
            except Exception as error: issues.append(f"标签异常:{label}:{error}")
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split = [paths for paths in duplicates if len({Path(path).parts[0] for path in paths}) > 1]
    return {"total": total, "split_counts": splits, "issues": issues, "exact_duplicate_groups": duplicates, "cross_split_exact_duplicate_groups": cross_split}


def write_manifest(plan: dict[str, Any], record_dir: Path, mode: str) -> None:
    record_dir.mkdir(parents=True, exist_ok=True); plan["mode"] = mode
    (record_dir / "import_manifest.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path); parser.add_argument("--pose-root", required=True, type=Path)
    parser.add_argument("--corner-root", required=True, type=Path); parser.add_argument("--quarantine-dir", required=True, type=Path)
    parser.add_argument("--record-dir", required=True, type=Path); parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(); plan = build_plan(args.analysis, args.pose_root.resolve(), args.corner_root.resolve())
    if args.apply:
        apply_plan(plan, args.quarantine_dir.resolve())
        plan["audit_after"] = {task: audit_dataset(Path(plan[key]), task) for task, key in TASK_ROOT_KEYS.items()}
        failed = any(audit["issues"] or audit["exact_duplicate_groups"] for audit in plan["audit_after"].values())
        write_manifest(plan, args.record_dir.resolve(), "apply_verified" if not failed else "apply_verification_failed")
        if failed: raise RuntimeError("导入后全量验证失败")
    else: write_manifest(plan, args.record_dir.resolve(), "dry_run")
    print(json.dumps({"mode": plan["mode"], "additions": dict(Counter(x["task"] for x in plan["additions"])), "removals": len(plan["removals"]), "replacements": len(plan["replacements"]), "audit_after": plan.get("audit_after")}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
