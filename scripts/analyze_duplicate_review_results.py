#!/usr/bin/env python3
"""Assess manually reviewed duplicate annotations for net-new training samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from build_task_training_manifests import read_annotation
from import_e_drive_training_data import (
    assignment_matches,
    corner_yolo,
    read_assignment_patient_ids,
    six_lr_pattern,
    six_point_yolo,
)


TASKS = ("six_point", "spine_pose")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_index(dataset: Path, cache_paths: list[Path] | None = None) -> dict[str, list[Path]]:
    cache_values: dict[str, str] = {}
    for cache_path in cache_paths or []:
        if not cache_path.is_file():
            continue
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            for path, metadata in loaded.items():
                if isinstance(metadata, dict) and isinstance(metadata.get("sha256"), str):
                    cache_values[str(Path(path).resolve())] = metadata["sha256"]
    index: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        for path in (dataset / "images" / split).glob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            digest = cache_values.get(str(path.resolve())) or sha256_file(path)
            index.setdefault(digest, []).append(path)
    return index


def parse_task_choices(row: dict[str, str], candidate_count: int) -> dict[str, int | None]:
    choice = row["选择"].strip()
    if choice.startswith("candidate:"):
        index = int(choice.split(":", 1)[1])
        if not 0 <= index < candidate_count:
            raise ValueError(f"{row['组号']}候选索引越界：{choice}")
        return {task: index for task in TASKS}
    if choice != "neither":
        raise ValueError(f"{row['组号']}未知选择：{choice}")
    note = row["备注"].strip()
    selected: dict[str, int | None] = {task: None for task in TASKS}
    task_words = {"six_point": r"(?:六点|6点)", "spine_pose": r"(?:椎体|脊柱)"}
    for task, word in task_words.items():
        match = re.search(rf"{word}[^，,；;。]*?(?:选|用)?图\s*([123])", note)
        if not match:
            match = re.search(rf"(?:选择)?图\s*([123])\s*的?{word}", note)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < candidate_count:
                selected[task] = index
    return selected


def resolve_path(candidate: dict[str, Any], kind: str) -> Path:
    name_key = "image" if kind == "image" else "annotation"
    source_key = f"{kind}_source"
    return Path(candidate[source_key]) / candidate[name_key]


def analyze(
    result_csv: Path,
    package_manifest: Path,
    assignment_xlsx: Path,
    pose_root: Path,
    corner_root: Path,
    hash_caches: list[Path] | None = None,
) -> dict[str, Any]:
    rows = list(csv.DictReader(result_csv.open(encoding="utf-8-sig", newline="")))
    manifest = json.loads(package_manifest.read_text(encoding="utf-8"))
    groups = {group["id"]: group for group in manifest["groups"]}
    if len(rows) != manifest["group_count"] or {row["组号"] for row in rows} != set(groups):
        raise ValueError("人工结果与核对包组数或组号不一致")
    if len({row["SHA256"] for row in rows}) != len(rows):
        raise ValueError("人工结果存在重复SHA256")
    patient_ids = read_assignment_patient_ids(assignment_xlsx)
    known = {
        "six_point": existing_index(pose_root, hash_caches),
        "spine_pose": existing_index(corner_root, hash_caches),
    }
    records: list[dict[str, Any]] = []
    for row in rows:
        group = groups[row["组号"]]
        if row["SHA256"] != group["sha256"]:
            raise ValueError(f"{row['组号']}的SHA256与manifest不一致")
        choices = parse_task_choices(row, len(group["candidates"]))
        for task in TASKS:
            index = choices[task]
            base = {
                "group": row["组号"],
                "sha256": row["SHA256"],
                "task": task,
                "review_choice": row["选择"],
                "note": row["备注"],
            }
            if index is None:
                if row["SHA256"] in known[task]:
                    records.append({**base, "status": "present_but_rejected", "reason": "人工未选可直接使用版本，但同图已在当前任务数据集中"})
                else:
                    records.append({**base, "status": "not_selected_for_task", "reason": "人工未为该任务选择可直接使用版本"})
                continue
            candidate = group["candidates"][index]
            image = resolve_path(candidate, "image")
            annotation = resolve_path(candidate, "annotation")
            record = {
                **base,
                "candidate_index": index,
                "source_image": str(image),
                "source_annotation": str(annotation),
            }
            matches = assignment_matches(image.name, patient_ids)
            if len(matches) != 1:
                reason = "不在assignment_all正式范围" if not matches else f"匹配多个患者ID：{matches}"
                records.append({**record, "status": "not_assignment", "reason": reason})
                continue
            if not image.is_file() or not annotation.is_file():
                records.append({**record, "status": "source_missing", "reason": "源图像或JSON不存在"})
                continue
            try:
                data = read_annotation(annotation)
                if task == "six_point":
                    pattern = six_lr_pattern(data)
                    label_text = six_point_yolo(data, lr_policy="preserve")
                    if pattern != "matches_target_CL_IL_SL_on_image_left":
                        raise ValueError(f"六点左右关系不符合当前规范：{pattern}")
                else:
                    label_text = corner_yolo(data)
            except Exception as error:  # noqa: BLE001 - every rejected record is reported
                records.append({**record, "status": "invalid_label", "reason": str(error)})
                continue
            digest = sha256_file(image)
            if digest != row["SHA256"]:
                records.append({**record, "status": "hash_mismatch", "reason": "源图哈希与重复组不一致"})
            elif digest in known[task]:
                existing_images = known[task][digest]
                existing_labels = [
                    image.parents[2] / "labels" / image.parent.name / f"{image.stem}.txt"
                    for image in existing_images
                ]
                same = [path for path in existing_labels if path.is_file() and path.read_text(encoding="utf-8") == label_text]
                if same:
                    records.append({**record, "status": "already_present_selected", "reason": "当前数据已有同图且标签与人工选择版本一致", "existing_labels": [str(path) for path in same]})
                else:
                    records.append({**record, "status": "needs_label_replacement", "reason": "当前数据已有同图，但标签不是人工选择版本", "existing_labels": [str(path) for path in existing_labels]})
            else:
                records.append({**record, "status": "eligible_new", "reason": "人工已选且通过正式范围、标签规范与内容去重检查"})
    summaries = {
        task: dict(Counter(record["status"] for record in records if record["task"] == task))
        for task in TASKS
    }
    return {
        "schema_version": 1,
        "result_csv": str(result_csv.resolve()),
        "package_manifest": str(package_manifest.resolve()),
        "review_groups": len(rows),
        "review_choices": dict(Counter(row["选择"] for row in rows)),
        "notes": sum(bool(row["备注"].strip()) for row in rows),
        "summary": summaries,
        "records": records,
    }


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 284组重复标注人工结果训练增量分析",
        "",
        f"人工结果覆盖{result['review_groups']}组；选择分布：{result['review_choices']}。",
        "",
        "| 任务 | 可净新增 | 已含所选版本 | 需替换现有标签 | 现有但被否决 | 未选且未入库 | 非正式范围 | 标签不合格 | 其他 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, label in (("six_point", "六点"), ("spine_pose", "角点")):
        summary = result["summary"][task]
        displayed = {"eligible_new", "already_present_selected", "needs_label_replacement", "present_but_rejected", "not_selected_for_task", "not_assignment", "invalid_label"}
        other = sum(value for key, value in summary.items() if key not in displayed)
        lines.append(
            f"| {label} | {summary.get('eligible_new', 0)} | {summary.get('already_present_selected', 0)} | "
            f"{summary.get('needs_label_replacement', 0)} | {summary.get('present_but_rejected', 0)} | "
            f"{summary.get('not_selected_for_task', 0)} | {summary.get('not_assignment', 0)} | "
            f"{summary.get('invalid_label', 0)} | {other} |"
        )
    lines.extend([
        "",
        "`eligible_new`仅表示可安全导入的净新增候选，本分析未修改训练数据。人工选择覆盖原先AI来源复核门槛，但仍强制要求assignment_all正式范围、任务点位完整、几何合法、左右规范一致且当前数据集无同内容图像。",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-csv", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--assignment-xlsx", required=True, type=Path)
    parser.add_argument("--pose-root", required=True, type=Path)
    parser.add_argument("--corner-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hash-cache", action="append", default=[], type=Path)
    args = parser.parse_args()
    result = analyze(args.result_csv, args.package_manifest, args.assignment_xlsx, args.pose_root, args.corner_root, args.hash_cache)
    write_report(result, args.output_dir)
    print(json.dumps({"review_groups": result["review_groups"], "summary": result["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
