#!/usr/bin/env python3
"""Import quality-approved normalized AP annotations into YOLO Pose train splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

from build_task_training_manifests import (
    VERTEBRA_NAMES,
    extract_task,
    fatal_reasons,
    read_annotation,
)


SIX_ORDER = ("CR", "CL", "IR", "IL", "SR", "SL")
CORNER_ORDER = (1, 2, 4, 3)  # JSON: TL/TR/BL/BR -> YOLO: TL/TR/BR/BL
LABEL_SUFFIX = "_label.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从规范化E盘数据按任务清单构建六点和椎体角点YOLO训练增量。"
    )
    parser.add_argument("export_dir", type=Path, help="规范化PNG和JSON目录")
    parser.add_argument("--manifest-dir", required=True, type=Path, help="任务独立训练清单目录")
    parser.add_argument("--pose-target", required=True, type=Path, help="六点YOLO数据集目录")
    parser.add_argument("--corner-target", required=True, type=Path, help="椎体角点YOLO数据集目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="导入报告与机器清单目录")
    parser.add_argument("--prefix", default="eap_", help="新增文件名前缀，默认eap_")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("six_point", "spine_pose"),
        default=("six_point", "spine_pose"),
        help="要构建的任务；默认同时处理六点和椎体角点",
    )
    parser.add_argument(
        "--six-lr-policy",
        choices=("block", "swap_pairs", "mirror_image"),
        default="block",
        help="六点左右约定冲突策略；默认block，另可交换三对标签或镜像图像并同步坐标",
    )
    parser.add_argument("--apply", action="store_true", help="实际复制；默认仅预演")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_trainable_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"判定", "图像文件", "标注文件", "重复组SHA256", "组内代表样本"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"清单字段不完整：{path}")
    return [row for row in rows if row["判定"] == "可训练"]


def verify_image(
    path: Path, data: dict[str, Any], validated_images: set[Path]
) -> tuple[int, int]:
    with Image.open(path) as image:
        size = image.size
        if path not in validated_images:
            image.load()
            validated_images.add(path)
    declared = (data.get("imageWidth"), data.get("imageHeight"))
    if declared != size:
        raise ValueError(f"图像尺寸与JSON不一致：{path.name} actual={size} json={declared}")
    return size


def bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    return ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)


def format_number(value: float) -> str:
    return f"{value:.8f}"


def six_point_yolo(
    data: dict[str, Any], *, accepted_anomaly: bool = False, lr_policy: str = "block"
) -> str:
    info = extract_task(data, "six_point", accepted_six_anomaly=accepted_anomaly)
    reasons = fatal_reasons(info, "六点")
    if reasons or info["geometry"]:
        raise ValueError("；".join([*reasons, *info["geometry"]]))
    if lr_policy == "swap_pairs":
        order = ("CL", "CR", "IL", "IR", "SL", "SR")
        points = [info["points"][label] for label in order]
    elif lr_policy == "mirror_image":
        points = [(1.0 - info["points"][label][0], info["points"][label][1]) for label in SIX_ORDER]
    else:
        points = [info["points"][label] for label in SIX_ORDER]
    values: list[str] = ["0", *(format_number(value) for value in bbox(points))]
    for x, y in points:
        values.extend((format_number(x), format_number(y), "2"))
    return " ".join(values) + "\n"


def six_lr_pattern(data: dict[str, Any], *, accepted_anomaly: bool = False) -> str:
    info = extract_task(data, "six_point", accepted_six_anomaly=accepted_anomaly)
    target_relations = [
        info["points"][right][0] < info["points"][left][0]
        for right, left in (("CR", "CL"), ("IR", "IL"), ("SR", "SL"))
    ]
    if all(target_relations):
        return "matches_target_CR_IR_SR_on_image_left"
    if not any(target_relations):
        return "opposite_to_target"
    return "mixed"


def corner_yolo(data: dict[str, Any]) -> str:
    info = extract_task(data, "spine_pose")
    reasons = fatal_reasons(info, "脊柱")
    if reasons or info["geometry"]:
        raise ValueError("；".join([*reasons, *info["geometry"]]))
    lines: list[str] = []
    for class_id, vertebra in enumerate(VERTEBRA_NAMES):
        points = [info["points"][f"{vertebra}-{index}"] for index in CORNER_ORDER]
        values: list[str] = [str(class_id), *(format_number(value) for value in bbox(points))]
        for x, y in points:
            values.extend((format_number(x), format_number(y), "2"))
        lines.append(" ".join(values))
    return "\n".join(lines) + "\n"


class HashCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: dict[str, dict[str, Any]] = {}
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.values = loaded
        self.pending = 0

    def hash(self, path: Path) -> str:
        resolved = str(path.resolve())
        stat = path.stat()
        cached = self.values.get(resolved)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            return str(cached["sha256"])
        value = sha256_file(path)
        self.values[resolved] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": value,
        }
        self.pending += 1
        if self.pending >= 16:
            self.flush()
        return value

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.values, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.pending = 0


def existing_hashes(dataset: Path, hash_file: Callable[[Path], str]) -> set[str]:
    hashes: set[str] = set()
    for split in ("train", "val", "test"):
        directory = dataset / "images" / split
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and not path.name.startswith("."):
                hashes.add(hash_file(path))
    return hashes


def plan_task(
    *,
    task: str,
    rows: list[dict[str, str]],
    export_dir: Path,
    target: Path,
    prefix: str,
    hash_file: Callable[[Path], str],
    accepted_six_anomalies: set[str],
    validated_images: set[Path],
    six_lr_policy: str,
) -> list[dict[str, Any]]:
    known_hashes = existing_hashes(target, hash_file)
    planned_hashes: set[str] = set()
    actions: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["标注文件"]):
        annotation = export_dir / row["标注文件"]
        image = export_dir / row["图像文件"]
        record: dict[str, Any] = {
            "task": task,
            "source_image": image.name,
            "source_annotation": annotation.name,
            "duplicate_group": row["重复组SHA256"],
            "duplicate_representative": row["组内代表样本"],
        }
        try:
            if not image.is_file() or not annotation.is_file():
                raise FileNotFoundError("源图像或JSON不存在")
            data = read_annotation(annotation)
            verify_image(image, data, validated_images)
            if task == "six_point":
                accepted_anomaly = annotation.name in accepted_six_anomalies
                lr_pattern = six_lr_pattern(data, accepted_anomaly=accepted_anomaly)
                label_text = six_point_yolo(
                    data,
                    accepted_anomaly=accepted_anomaly,
                    lr_policy=six_lr_policy,
                )
                record["six_lr_pattern"] = lr_pattern
                record["image_transform"] = (
                    "horizontal_mirror" if six_lr_policy == "mirror_image" else "none"
                )
            else:
                label_text = corner_yolo(data)
                record["image_transform"] = "none"
            image_hash = hash_file(image)
            destination_stem = f"{prefix}{image.stem}"
            destination_image = target / "images" / "train" / f"{destination_stem}{image.suffix.lower()}"
            destination_label = target / "labels" / "train" / f"{destination_stem}.txt"
            record.update(
                {
                    "source_image_sha256": image_hash,
                    "destination_image": str(destination_image),
                    "destination_label": str(destination_label),
                    "label_text": label_text,
                }
            )
            if (
                task == "six_point"
                and six_lr_policy in {"swap_pairs", "mirror_image"}
                and record["six_lr_pattern"] != "opposite_to_target"
            ):
                record.update(
                    status="skipped",
                    reason="六点左右关系不是可按整批策略自动统一的完全相反模式",
                )
            elif image_hash in known_hashes:
                record.update(status="skipped", reason="与目标数据集现有图像内容重复")
            elif image_hash in planned_hashes:
                record.update(status="skipped", reason="本次任务候选内部图像内容重复")
            elif destination_image.exists() or destination_label.exists():
                image_matches = False
                if destination_image.is_file():
                    image_matches = (
                        mirrored_pixels_equal(image, destination_image)
                        if record["image_transform"] == "horizontal_mirror"
                        else hash_file(destination_image) == image_hash
                    )
                if (
                    destination_image.is_file()
                    and destination_label.is_file()
                    and image_matches
                    and destination_label.read_text(encoding="utf-8") == label_text
                ):
                    record.update(status="already_imported", reason="目标文件已存在且内容一致")
                else:
                    record.update(status="error", reason="目标文件名冲突且内容不一致")
            else:
                record.update(status="planned", reason="通过任务质量与去重检查")
                planned_hashes.add(image_hash)
        except Exception as exc:  # noqa: BLE001 - each sample must be reported
            record.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        actions.append(record)
    return actions


def mirrored_pixels_equal(source: Path, destination: Path) -> bool:
    with Image.open(source) as source_image, Image.open(destination) as destination_image:
        mirrored = ImageOps.mirror(source_image)
        if mirrored.mode != destination_image.mode or mirrored.size != destination_image.size:
            return False
        return mirrored.tobytes() == destination_image.tobytes()


def apply_actions(
    actions: list[dict[str, Any]], export_dir: Path, hash_file: Callable[[Path], str]
) -> None:
    for action in actions:
        if action["status"] != "planned":
            continue
        source = export_dir / action["source_image"]
        image_target = Path(action["destination_image"])
        label_target = Path(action["destination_label"])
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        temporary_target = image_target.with_name(f".{image_target.name}.tmp")
        if action.get("image_transform") == "horizontal_mirror":
            with Image.open(source) as source_image:
                ImageOps.mirror(source_image).save(temporary_target, format="PNG")
            if not mirrored_pixels_equal(source, temporary_target):
                temporary_target.unlink(missing_ok=True)
                raise RuntimeError(f"镜像后像素核验失败：{source.name}")
        else:
            shutil.copy2(source, temporary_target)
        if action.get("image_transform") != "horizontal_mirror" and hash_file(source) != hash_file(temporary_target):
            temporary_target.unlink(missing_ok=True)
            raise RuntimeError(f"复制后SHA-256不一致：{source.name}")
        temporary_target.replace(image_target)
        temporary_label = label_target.with_name(f".{label_target.name}.tmp")
        temporary_label.write_text(action["label_text"], encoding="utf-8")
        temporary_label.replace(label_target)
        action["destination_image_sha256"] = hash_file(image_target)
        action["status"] = "imported"
        action["reason"] = "已复制到train并通过SHA-256验证"


def public_action(action: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in action.items() if key != "label_text"}


def build(
    export_dir: Path,
    manifest_dir: Path,
    pose_target: Path,
    corner_target: Path,
    output_dir: Path,
    *,
    prefix: str = "eap_",
    apply: bool = False,
    six_lr_policy: str = "block",
    tasks: tuple[str, ...] = ("six_point", "spine_pose"),
) -> dict[str, Any]:
    requested_apply = apply
    paths = [export_dir, manifest_dir, pose_target, corner_target]
    if not all(path.expanduser().is_dir() for path in paths):
        raise NotADirectoryError("源目录、清单目录或目标数据集目录不存在")
    export_dir = export_dir.expanduser().resolve()
    manifest_dir = manifest_dir.expanduser().resolve()
    pose_target = pose_target.expanduser().resolve()
    corner_target = corner_target.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not prefix or "/" in prefix or "\\" in prefix:
        raise ValueError("prefix不能为空且不能包含路径分隔符")
    if six_lr_policy not in {"block", "swap_pairs", "mirror_image"}:
        raise ValueError("six_lr_policy必须为block、swap_pairs或mirror_image")
    selected_tasks = tuple(dict.fromkeys(tasks))
    if not selected_tasks or not set(selected_tasks).issubset({"six_point", "spine_pose"}):
        raise ValueError("tasks必须从six_point和spine_pose中选择")
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_cache = HashCache(output_dir / "sha256_cache.json")
    summary_path = manifest_dir / "清单汇总.json"
    accepted_six_anomalies: set[str] = set()
    if summary_path.is_file():
        manifest_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        accepted_six_anomalies = set(
            manifest_summary.get("rules", {}).get("accepted_six_anomalies", [])
        )

    task_specs = {
        "six_point": ("六点模型样本清单.csv", pose_target),
        "spine_pose": ("脊柱Pose模型样本清单.csv", corner_target),
    }
    actions: dict[str, list[dict[str, Any]]] = {}
    validated_images: set[Path] = set()
    for task in selected_tasks:
        manifest_name, target = task_specs[task]
        rows = read_trainable_rows(manifest_dir / manifest_name)
        actions[task] = plan_task(
            task=task,
            rows=rows,
            export_dir=export_dir,
            target=target,
            prefix=prefix,
            hash_file=hash_cache.hash,
            accepted_six_anomalies=accepted_six_anomalies,
            validated_images=validated_images,
            six_lr_policy=six_lr_policy,
        )
        hash_cache.flush()
    lr_counts = Counter(
        action.get("six_lr_pattern", "not_checked")
        for action in actions.get("six_point", [])
    )
    compatibility_issues: list[str] = []
    if any(
        pattern not in {"matches_target_CR_IR_SR_on_image_left", "not_checked"}
        for pattern in lr_counts
    ):
        compatibility_issues.append("E盘六点左右语义约定与现有pose_data不一致")
    blocked_reasons: list[str] = []
    if apply and six_lr_policy == "block":
        blocked_reasons.extend(compatibility_issues)
    if any(action["status"] == "error" for task_actions in actions.values() for action in task_actions):
        blocked_reasons.append("存在样本级校验错误")
    if blocked_reasons:
        apply = False
    if apply:
        for task_actions in actions.values():
            apply_actions(task_actions, export_dir, hash_cache.hash)
        hash_cache.flush()

    summary = {
        task: {
            "manifest_trainable": len(task_actions),
            "statuses": dict(sorted(Counter(action["status"] for action in task_actions).items())),
            "duplicate_representatives": sum(
                action["duplicate_representative"] == "是" for action in task_actions
            ),
        }
        for task, task_actions in actions.items()
    }
    result = {
        "schema_version": 1,
        "requested_mode": "apply" if requested_apply else "dry_run",
        "mode": "apply" if apply else "dry_run",
        "export_dir": str(export_dir),
        "manifest_dir": str(manifest_dir),
        "prefix": prefix,
        "tasks": list(selected_tasks),
        "six_lr_policy": six_lr_policy,
        "six_lr_patterns": dict(sorted(lr_counts.items())),
        "compatibility_issues": compatibility_issues,
        "blocked_reasons": blocked_reasons,
        "summary": summary,
        "actions": {
            task: [public_action(action) for action in task_actions]
            for task, task_actions in actions.items()
        },
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# E盘新增AP数据训练集构建报告

模式：{'实际导入' if apply else '预演，未修改目标数据集'}

## 构建规则

- 数据源：`{export_dir}`
- 仅读取任务清单中判定为“可训练”的样本。
- 六点与椎体任务独立选择，因此新增数量允许不同。
- 重复组只采用清单已选出的任务代表样本；未裁决的冲突组不导入。
- 对源图像做完整解码和JSON尺寸核对，对转换后的任务标签重做完整性、坐标和几何检查。
- 与目标数据集任一train/val/test图像内容重复的样本跳过，避免训练或评估泄漏。
- 新样本只追加到train，既有val/test保持不变。
- 六点左右约定策略：`{six_lr_policy}`；候选关系统计：`{dict(sorted(lr_counts.items()))}`。
- 兼容性问题：`{compatibility_issues or '无'}`。
- 阻止实际导入的原因：`{blocked_reasons or '无'}`。

## 结果

- 六点：{summary.get('six_point', '本次未处理')}
- 椎体角点：{summary.get('spine_pose', '本次未处理')}

逐样本来源、SHA-256、目标路径和跳过原因见 `import_manifest.json`。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(
        args.export_dir,
        args.manifest_dir,
        args.pose_target,
        args.corner_target,
        args.output_dir,
        prefix=args.prefix,
        apply=args.apply,
        six_lr_policy=args.six_lr_policy,
        tasks=tuple(args.tasks),
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    has_errors = any("error" in summary["statuses"] for summary in result["summary"].values())
    return int(has_errors or bool(result["blocked_reasons"]))


if __name__ == "__main__":
    raise SystemExit(main())
