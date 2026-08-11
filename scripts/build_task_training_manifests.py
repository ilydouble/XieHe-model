#!/usr/bin/env python3
"""Build independent six-point and spine-pose training candidate manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

LABEL_SUFFIX = "_label.json"
SIX_POINT_LABELS = ("CL", "CR", "IL", "IR", "SL", "SR")
VERTEBRA_NAMES = ("C7",) + tuple(f"T{i}" for i in range(1, 13)) + tuple(
    f"L{i}" for i in range(1, 6)
)
SPINE_LABELS = tuple(
    f"{vertebra}-{corner}"
    for vertebra in VERTEBRA_NAMES
    for corner in range(1, 5)
)
DECISION_TRAIN = "可训练"
DECISION_REVIEW = "待复核"
DECISION_EXCLUDE = "排除"
CSV_FIELDS = (
    "任务",
    "判定",
    "图像文件",
    "标注文件",
    "原因",
    "来源概况",
    "完整点数",
    "期望点数",
    "重复组SHA256",
    "重复组图像数",
    "组内最大归一化差异",
    "组内代表样本",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为六点与脊柱 Pose 独立生成训练分级清单。")
    parser.add_argument("export_dir", type=Path, help="规范化图像及 JSON 所在目录")
    parser.add_argument("--audit", required=True, type=Path, help="全量审计 JSON")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.005,
        help="重复版本可自动选代表样本的最大归一化点距，默认 0.005",
    )
    parser.add_argument(
        "--accepted-six-anomaly",
        action="append",
        default=[],
        metavar="ANNOTATION_JSON",
        help="人工确认可接受的六点左右关系例外，可重复传入",
    )
    return parser.parse_args(argv)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def annotation_name_for_image(image_name: str) -> str:
    return f"{Path(image_name).stem}{LABEL_SUFFIX}"


def image_name_for_annotation(annotation_name: str) -> str:
    if not annotation_name.endswith(LABEL_SUFFIX):
        raise ValueError(f"标注文件名不符合约定：{annotation_name}")
    return f"{annotation_name[:-len(LABEL_SUFFIX)]}.png"


def read_annotation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层不是对象：{path.name}")
    return data


def extract_task(
    data: dict[str, Any], task: str, accepted_six_anomaly: bool = False
) -> dict[str, Any]:
    expected = SIX_POINT_LABELS if task == "six_point" else SPINE_LABELS
    expected_set = set(expected)
    points: dict[str, tuple[float, float]] = {}
    sources: dict[str, str] = {}
    duplicates: set[str] = set()
    invalid: set[str] = set()
    out_of_range: set[str] = set()

    items = data.get("vertebrae")
    if not isinstance(items, list):
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if label not in expected_set:
            continue
        if label in points or label in invalid:
            duplicates.add(label)
        coordinate: object
        if task == "six_point":
            coordinate = item.get("point")
        else:
            corners = item.get("corners")
            coordinate = corners[0] if isinstance(corners, list) and corners else None
            if not isinstance(corners, list) or len(corners) != 4:
                invalid.add(label)
        if not isinstance(coordinate, dict):
            invalid.add(label)
            continue
        x, y = coordinate.get("x"), coordinate.get("y")
        if not is_number(x) or not is_number(y):
            invalid.add(label)
            continue
        x_value, y_value = float(x), float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            invalid.add(label)
            continue
        points[label] = (x_value, y_value)
        sources[label] = str(item.get("source", "unknown"))
        if not 0 <= x_value <= 1 or not 0 <= y_value <= 1:
            out_of_range.add(label)

    missing = expected_set - set(points)
    geometry: list[str] = []
    if task == "six_point" and not accepted_six_anomaly:
        for left, right in (("CL", "CR"), ("IL", "IR"), ("SL", "SR")):
            if left in points and right in points and points[left][0] >= points[right][0]:
                geometry.append(f"{left}/{right}左右关系")
    if task == "spine_pose":
        for vertebra in VERTEBRA_NAMES:
            labels = tuple(f"{vertebra}-{corner}" for corner in range(1, 5))
            if not all(label in points for label in labels):
                continue
            p1, p2, p3, p4 = (points[label] for label in labels)
            if p1[0] >= p2[0] or p3[0] >= p4[0]:
                geometry.append(f"{vertebra}左右角点顺序")
            if (p1[1] + p2[1]) / 2 >= (p3[1] + p4[1]) / 2:
                geometry.append(f"{vertebra}上下角点顺序")

    source_values = [sources.get(label, "missing") for label in expected]
    manual_count = source_values.count("manual")
    ai_count = source_values.count("ai")
    if manual_count == len(expected):
        source_profile = "全manual"
    elif ai_count == len(expected):
        source_profile = "全AI"
    else:
        source_profile = f"mixed(manual={manual_count},ai={ai_count},其他={len(expected)-manual_count-ai_count})"

    return {
        "points": points,
        "sources": sources,
        "missing": sorted(missing),
        "duplicates": sorted(duplicates),
        "invalid": sorted(invalid),
        "out_of_range": sorted(out_of_range),
        "geometry": geometry,
        "source_profile": source_profile,
        "manual_count": manual_count,
        "ai_count": ai_count,
        "complete_count": len(expected_set & set(points)),
        "expected_count": len(expected),
    }


def fatal_reasons(info: dict[str, Any], task_label: str) -> list[str]:
    reasons: list[str] = []
    if info["missing"]:
        reasons.append(f"缺少{len(info['missing'])}个{task_label}标准点")
    if info["duplicates"]:
        reasons.append(f"标签重复:{'|'.join(info['duplicates'])}")
    if info["invalid"]:
        reasons.append(f"坐标结构非法:{'|'.join(info['invalid'])}")
    if info["out_of_range"]:
        reasons.append(f"坐标越界:{'|'.join(info['out_of_range'])}")
    return reasons


def source_eligible(info: dict[str, Any], task: str) -> bool:
    if task == "six_point":
        return info["manual_count"] == info["expected_count"]
    return info["manual_count"] > 0


def source_review_reason(info: dict[str, Any], task: str) -> str:
    if task == "six_point":
        return "六点并非全部manual"
    return "脊柱72点均为AI，尚无人工修正证据"


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.dist(left, right)


def pair_distance(left: dict[str, Any], right: dict[str, Any], labels: Iterable[str]) -> float:
    distances = [distance(left["points"][label], right["points"][label]) for label in labels]
    return max(distances, default=0.0)


def medoid_name(
    names: list[str], infos: dict[str, dict[str, Any]], labels: Iterable[str]
) -> str:
    scores = {}
    for name in names:
        scores[name] = sum(
            pair_distance(infos[name], infos[other], labels)
            for other in names
            if other != name
        )
    return min(names, key=lambda name: (scores[name], name))


def make_row(
    *,
    task_name: str,
    image_name: str,
    annotation_name: str,
    decision: str,
    reason: str,
    info: dict[str, Any],
    duplicate_sha: str = "",
    duplicate_size: int = 1,
    max_difference: float | None = None,
    representative: bool = False,
) -> dict[str, str | int]:
    return {
        "任务": task_name,
        "判定": decision,
        "图像文件": image_name,
        "标注文件": annotation_name,
        "原因": reason,
        "来源概况": info["source_profile"],
        "完整点数": info["complete_count"],
        "期望点数": info["expected_count"],
        "重复组SHA256": duplicate_sha,
        "重复组图像数": duplicate_size,
        "组内最大归一化差异": "" if max_difference is None else f"{max_difference:.8f}",
        "组内代表样本": "是" if representative else "否",
    }


def classify_task(
    *,
    task: str,
    annotation_data: dict[str, dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
    threshold: float,
    accepted_six_anomalies: set[str],
) -> list[dict[str, str | int]]:
    task_name = "六点模型" if task == "six_point" else "脊柱Pose模型"
    task_label = "六点" if task == "six_point" else "脊柱"
    expected = SIX_POINT_LABELS if task == "six_point" else SPINE_LABELS
    infos = {
        name: extract_task(
            data,
            task,
            accepted_six_anomaly=task == "six_point" and name in accepted_six_anomalies,
        )
        for name, data in annotation_data.items()
    }
    duplicate_by_annotation: dict[str, dict[str, Any]] = {}
    for group in duplicate_groups:
        for image_name in group.get("files", []):
            annotation_name = annotation_name_for_image(image_name)
            if annotation_name in annotation_data:
                duplicate_by_annotation[annotation_name] = group

    rows: list[dict[str, str | int]] = []
    processed: set[str] = set()
    for annotation_name in sorted(annotation_data):
        if annotation_name in processed:
            continue
        group = duplicate_by_annotation.get(annotation_name)
        if group is None:
            info = infos[annotation_name]
            fatal = fatal_reasons(info, task_label)
            if fatal:
                decision, reason = DECISION_EXCLUDE, "；".join(fatal)
            elif info["geometry"]:
                decision, reason = DECISION_REVIEW, "；".join(info["geometry"])
            elif source_eligible(info, task):
                decision, reason = DECISION_TRAIN, "任务标注完整、坐标与结构检查通过"
            else:
                decision, reason = DECISION_REVIEW, source_review_reason(info, task)
            rows.append(
                make_row(
                    task_name=task_name,
                    image_name=image_name_for_annotation(annotation_name),
                    annotation_name=annotation_name,
                    decision=decision,
                    reason=reason,
                    info=info,
                )
            )
            processed.add(annotation_name)
            continue

        names = sorted(
            annotation_name_for_image(image_name)
            for image_name in group.get("files", [])
            if annotation_name_for_image(image_name) in annotation_data
        )
        processed.update(names)
        valid_names = [name for name in names if not fatal_reasons(infos[name], task_label)]
        eligible_names = [
            name
            for name in valid_names
            if not infos[name]["geometry"] and source_eligible(infos[name], task)
        ]
        max_difference: float | None = None
        representative: str | None = None
        if len(valid_names) >= 2:
            max_difference = max(
                pair_distance(infos[left], infos[right], expected)
                for index, left in enumerate(valid_names)
                for right in valid_names[index + 1 :]
            )
            if (
                len(eligible_names) == len(valid_names)
                and max_difference <= threshold
            ):
                representative = medoid_name(valid_names, infos, expected)
        elif len(valid_names) == 1 and valid_names == eligible_names:
            representative = valid_names[0]

        for name in names:
            info = infos[name]
            fatal = fatal_reasons(info, task_label)
            if fatal:
                decision = DECISION_EXCLUDE
                reason = "；".join(fatal) + "；同图存在其他标注版本"
            elif representative is not None and name == representative:
                decision = DECISION_TRAIN
                if len(valid_names) == 1:
                    reason = "重复组仅此版本满足任务质量规则，选为代表样本"
                else:
                    reason = f"重复组差异≤{threshold:g}，选取medoid代表样本"
            elif representative is not None:
                decision = DECISION_EXCLUDE
                reason = f"同图冗余副本；代表样本为{representative}"
            elif info["geometry"]:
                decision = DECISION_REVIEW
                reason = "；".join(info["geometry"]) + "；重复组需人工裁决"
            elif not source_eligible(info, task):
                decision = DECISION_REVIEW
                reason = source_review_reason(info, task) + "；重复组需人工裁决"
            else:
                decision = DECISION_REVIEW
                diff_text = "结构不一致" if max_difference is None else f"最大差异{max_difference:.8f}"
                reason = f"同图任务标注冲突（{diff_text}），需人工裁决"
            rows.append(
                make_row(
                    task_name=task_name,
                    image_name=image_name_for_annotation(name),
                    annotation_name=name,
                    decision=decision,
                    reason=reason,
                    info=info,
                    duplicate_sha=str(group.get("sha256", "")),
                    duplicate_size=len(group.get("files", [])),
                    max_difference=max_difference,
                    representative=name == representative,
                )
            )
    return sorted(rows, key=lambda row: (str(row["判定"]), str(row["图像文件"])))


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str | int]]) -> dict[str, Any]:
    decisions = Counter(str(row["判定"]) for row in rows)
    duplicate_decisions = Counter(
        str(row["判定"]) for row in rows if row["重复组SHA256"]
    )
    return {
        "total": len(rows),
        "decisions": dict(sorted(decisions.items())),
        "duplicate_rows": sum(1 for row in rows if row["重复组SHA256"]),
        "duplicate_decisions": dict(sorted(duplicate_decisions.items())),
        "selected_duplicate_representatives": sum(
            1 for row in rows if row["重复组SHA256"] and row["组内代表样本"] == "是"
        ),
    }


def build_manifests(
    export_dir: Path,
    audit_path: Path,
    output_dir: Path,
    *,
    threshold: float = 0.005,
    accepted_six_anomalies: Iterable[str] = (),
) -> dict[str, Any]:
    export_dir = export_dir.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not export_dir.is_dir():
        raise NotADirectoryError(export_dir)
    if threshold < 0:
        raise ValueError("duplicate threshold 不能为负数")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    annotation_paths = sorted(export_dir.glob(f"*{LABEL_SUFFIX}"))
    annotation_data = {path.name: read_annotation(path) for path in annotation_paths}
    duplicate_groups = audit.get("exact_duplicate_groups")
    if not isinstance(duplicate_groups, list):
        raise ValueError("审计 JSON 缺少 exact_duplicate_groups")

    accepted = set(accepted_six_anomalies)
    six_rows = classify_task(
        task="six_point",
        annotation_data=annotation_data,
        duplicate_groups=duplicate_groups,
        threshold=threshold,
        accepted_six_anomalies=accepted,
    )
    spine_rows = classify_task(
        task="spine_pose",
        annotation_data=annotation_data,
        duplicate_groups=duplicate_groups,
        threshold=threshold,
        accepted_six_anomalies=accepted,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    six_path = output_dir / "六点模型样本清单.csv"
    spine_path = output_dir / "脊柱Pose模型样本清单.csv"
    write_csv(six_path, six_rows)
    write_csv(spine_path, spine_rows)

    summary = {
        "schema_version": 1,
        "export_dir": str(export_dir),
        "audit": str(audit_path),
        "rules": {
            "duplicate_threshold": threshold,
            "six_point_train_source": "6/6 manual",
            "spine_pose_train_source": "72点中至少1点manual；全AI待复核",
            "duplicate_selection": "任务内全部有效版本满足来源/结构规则且最大点距不超过阈值时，选medoid；其余待复核",
            "accepted_six_anomalies": sorted(accepted),
        },
        "six_point": summarize(six_rows),
        "spine_pose": summarize(spine_rows),
    }
    (output_dir / "清单汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    explanation = f"""# 两个模型独立训练样本清单说明

数据源：`{export_dir}`

本目录只包含分级清单，不移动、不删除、不修改图像或 JSON。

## 判定含义

- `可训练`：满足当前任务的完整性、坐标、结构和来源规则；重复组只选一份代表样本。
- `待复核`：标注结构完整，但来源证据不足、存在任务内重复冲突或几何关系需要人工确认。
- `排除`：任务标注缺失、坐标非法/越界，或属于已经选出代表样本的同图冗余副本。

## 当前规则

- 六点模型：六点必须完整且全部为 `manual`；已人工确认的方向例外可显式放行。
- 脊柱 Pose：72 点必须完整且角点顺序正常；允许 `manual`/`ai` 混合，全 AI 标注进入待复核。
- 重复组选优阈值：最大归一化点距 ≤ `{threshold:g}`；满足任务质量规则后以 medoid 选一个代表版本。
- 两个模型独立判定，同一图可以只进入其中一个模型。
- 清单尚未划分 train/val/test；后续必须按患者/影像组拆分，并预留两个模型共同的端到端测试患者。

## 汇总

六点模型：{summary['six_point']['decisions']}

脊柱 Pose：{summary['spine_pose']['decisions']}
"""
    (output_dir / "清单说明.md").write_text(explanation, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_manifests(
        args.export_dir,
        args.audit,
        args.output_dir,
        threshold=args.duplicate_threshold,
        accepted_six_anomalies=args.accepted_six_anomaly,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
