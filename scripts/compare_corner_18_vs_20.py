#!/usr/bin/env python3
"""Compare an 18-class Corner checkpoint with a 20-class checkpoint fairly."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps

import compare_corner_models as corner_eval


BASE_CLASS_IDS = frozenset(range(18))
EXTRA_CLASS_IDS = frozenset({18, 19})


def filter_classes(objects: dict[int, corner_eval.CornerObject], classes: frozenset[int]) -> dict[int, corner_eval.CornerObject]:
    return {class_id: item for class_id, item in objects.items() if class_id in classes}


def bootstrap_mean_ci(values: Sequence[float], seed: int = 20260825, iterations: int = 5000) -> list[float] | None:
    if not values:
        return None
    generator = random.Random(seed)
    means = []
    for _ in range(iterations):
        means.append(statistics.fmean(generator.choice(values) for _ in values))
    means.sort()
    return [round(means[round((len(means) - 1) * fraction)], 4) for fraction in (0.025, 0.975)]


def format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def aggregate(samples: Sequence[dict], sample_key: str) -> dict:
    adapted = [{"model": {"native": sample[sample_key]}} for sample in samples]
    return corner_eval.aggregate_mode(adapted, "model", "native")


def aggregate_sources(samples: Sequence[dict]) -> dict:
    output = {}
    for source in sorted({corner_eval.source_group(sample["filename"]) for sample in samples}):
        rows = [sample for sample in samples if corner_eval.source_group(sample["filename"]) == source]
        old_errors = [sample["old_base"]["mean_error_px"] for sample in rows]
        new_errors = [sample["new_base"]["mean_error_px"] for sample in rows]
        deltas = [old - new for old, new in zip(old_errors, new_errors)]
        output[source] = {
            "images": len(rows),
            "old_mean_image_error_px": round(statistics.fmean(old_errors), 4),
            "new_mean_image_error_px": round(statistics.fmean(new_errors), 4),
            "mean_image_improvement_px": round(statistics.fmean(deltas), 4),
            "improved_images": sum(value > 0.05 for value in deltas),
            "worsened_images": sum(value < -0.05 for value in deltas),
        }
    return output


def summarize_extra_predictions(samples: Sequence[dict]) -> dict:
    rare_samples = [sample for sample in samples if sample["new_extra"] is not None]
    rare_summary = aggregate(rare_samples, "new_extra")
    per_class = {}
    for class_id, semantic in ((18, "V18_L6"), (19, "V19_T13")):
        truth_count = sum(class_id in sample["extra_truth_classes"] for sample in samples)
        predicted_count = sum(class_id in sample["new_extra_predicted_classes"] for sample in samples)
        matched_count = sum(
            class_id in sample["extra_truth_classes"] and class_id in sample["new_extra_predicted_classes"]
            for sample in samples
        )
        rows = [
            row
            for sample in rare_samples
            for row in sample["new_extra"]["point_rows"]
            if row["class_id"] == class_id and row["visible"]
        ]
        errors = [row["distance_px"] for row in rows if row["distance_px"] is not None]
        per_class[semantic] = {
            "truth_vertebrae": truth_count,
            "predicted_vertebrae": predicted_count,
            "detected_vertebrae": matched_count,
            "false_positive_vertebrae": predicted_count - matched_count,
            "precision": matched_count / predicted_count if predicted_count else None,
            "visible_points": len(rows),
            "point_recall": len(errors) / len(rows) if rows else None,
            "mean_error_px": None if not errors else round(statistics.fmean(errors), 3),
        }
    truth_total = sum(len(sample["extra_truth_classes"]) for sample in samples)
    predicted_total = sum(len(sample["new_extra_predicted_classes"]) for sample in samples)
    matched_total = sum(
        len(set(sample["extra_truth_classes"]) & set(sample["new_extra_predicted_classes"]))
        for sample in samples
    )
    return {
        "images": len(rare_samples),
        "truth_vertebrae": truth_total,
        "predicted_vertebrae": predicted_total,
        "detected_vertebrae": matched_total,
        "false_positive_vertebrae": predicted_total - matched_total,
        "false_positive_images": sum(
            bool(set(sample["new_extra_predicted_classes"]) - set(sample["extra_truth_classes"]))
            for sample in samples
        ),
        "precision": matched_total / predicted_total if predicted_total else None,
        "visible_points": rare_summary["visible_points"],
        "detected_points": rare_summary["detected_points"],
        "point_recall": rare_summary["point_recall"],
        "mean_error_px": rare_summary["mean_error_px"],
        "pck_20_all": rare_summary["pck_20_all"],
        "per_class": per_class,
    }


def select_representatives(samples: Sequence[dict], count_each: int = 4) -> list[dict]:
    ordered = sorted(samples, key=lambda sample: sample["base_improvement_px"])
    middle = len(ordered) // 2
    groups = (
        ("worse", ordered[:count_each]),
        ("median", ordered[max(0, middle - count_each // 2): middle + (count_each + 1) // 2]),
        ("better", reversed(ordered[-count_each:])),
        ("extra", (sample for sample in samples if sample["extra_truth_classes"])),
    )
    selected = []
    seen = set()
    for group, rows in groups:
        for sample in rows:
            if sample["filename"] in seen:
                continue
            seen.add(sample["filename"])
            selected.append({"group": group, **sample})
    return selected


def render_preview(
    image_path: Path,
    truth: dict[int, corner_eval.CornerObject],
    old_prediction: dict[int, corner_eval.CornerObject],
    new_prediction: dict[int, corner_eval.CornerObject],
    old_error: float | None,
    new_error: float | None,
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        source.load()
        source = ImageOps.exif_transpose(source).convert("RGB")
        base, scale = corner_eval.fit_image(source)
    gt = corner_eval.add_title(
        corner_eval.draw_objects(base, truth.values(), corner_eval.GT_COLOR, scale),
        f"GT: {len(truth)} classes",
        corner_eval.GT_COLOR,
    )
    old = corner_eval.draw_objects(base, truth.values(), corner_eval.GT_COLOR, scale)
    old = corner_eval.add_title(
        corner_eval.draw_objects(old, old_prediction.values(), corner_eval.OLD_COLOR, scale),
        f"Old 18c base: {old_error} px",
        corner_eval.OLD_COLOR,
    )
    new = corner_eval.draw_objects(base, truth.values(), corner_eval.GT_COLOR, scale)
    new = corner_eval.add_title(
        corner_eval.draw_objects(new, new_prediction.values(), corner_eval.NEW_COLOR, scale),
        f"New 20c base: {new_error} px",
        corner_eval.NEW_COLOR,
    )
    gap = 8
    canvas = Image.new("RGB", (gt.width * 3 + gap * 2, gt.height), (70, 70, 70))
    canvas.paste(gt, (0, 0))
    canvas.paste(old, (gt.width + gap, 0))
    canvas.paste(new, ((gt.width + gap) * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=88, optimize=True)


def write_csv(path: Path, samples: Sequence[dict]) -> None:
    fields = (
        "filename", "source", "extra_truth_classes", "old_base_error_px", "new_base_error_px",
        "base_improvement_px", "old_base_recall", "new_base_recall", "old_base_predicted",
        "new_base_predicted", "new_extra_detected_points", "new_extra_visible_points", "old_ms", "new_ms",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            extra = sample["new_extra"]
            writer.writerow({
                "filename": sample["filename"],
                "source": corner_eval.source_group(sample["filename"]),
                "extra_truth_classes": ",".join(map(str, sample["extra_truth_classes"])),
                "old_base_error_px": sample["old_base"]["mean_error_px"],
                "new_base_error_px": sample["new_base"]["mean_error_px"],
                "base_improvement_px": sample["base_improvement_px"],
                "old_base_recall": sample["old_base"]["detected_points"] / sample["old_base"]["visible_points"],
                "new_base_recall": sample["new_base"]["detected_points"] / sample["new_base"]["visible_points"],
                "old_base_predicted": sample["old_base"]["predicted_vertebrae"],
                "new_base_predicted": sample["new_base"]["predicted_vertebrae"],
                "new_extra_detected_points": None if extra is None else extra["detected_points"],
                "new_extra_visible_points": None if extra is None else extra["visible_points"],
                "old_ms": sample["old_ms"],
                "new_ms": sample["new_ms"],
            })


def automatic_interpretation(old: dict, new: dict) -> str:
    error_gain = old["mean_error_px"] - new["mean_error_px"]
    relative_gain = error_gain / old["mean_error_px"]
    recall_gain = new["point_recall"] - old["point_recall"]
    pck_gain = new["pck_20_all"] - old["pck_20_all"]
    if relative_gain >= 0.03 and recall_gain >= -0.002 and pck_gain > 0:
        return "V0–V17基础能力有实质提高。"
    if relative_gain > 0 and pck_gain >= 0 and recall_gain >= -0.005:
        return "V0–V17有轻度提高，但幅度有限。"
    if abs(relative_gain) < 0.01 and abs(pck_gain) < 0.003:
        return "V0–V17基本持平。"
    return "V0–V17没有形成一致提高，不能仅因支持新增类别就替换旧模型。"


def build_interpretation(old: dict, new: dict, comparison: dict, sources: dict, extra: dict) -> str:
    parts = [automatic_interpretation(old, new)]
    confidence_interval = comparison["mean_image_improvement_95ci_px"]
    worsened_sources = [name for name, values in sources.items() if values["mean_image_improvement_px"] < 0]
    if confidence_interval[0] <= 0 <= confidence_interval[1]:
        parts.append("但逐图平均改善的95% bootstrap区间跨0，收益不够稳定。")
    if worsened_sources:
        parts.append(f"改善主要由部分来源驱动，{', '.join(worsened_sources)}的平均误差反而上升。")
    if extra["detected_vertebrae"] < extra["truth_vertebrae"]:
        parts.append(
            f"新增V18/V19只检出{extra['detected_vertebrae']}/{extra['truth_vertebrae']}个，20类能力尚未达到可用水平。"
        )
    return "".join(parts)


def write_report(path: Path, manifest: dict) -> None:
    old = manifest["base_summary"]["old_18class"]
    new = manifest["base_summary"]["new_20class"]
    comparison = manifest["base_comparison"]
    extra = manifest["extra_summary"]
    vertebra_rows = []
    for class_id in range(18):
        name = f"V{class_id}"
        old_error = old["per_vertebra"][name]["mean_error_px"]
        new_error = new["per_vertebra"][name]["mean_error_px"]
        vertebra_rows.append(f"| {name} | {old_error} | {new_error} | {old_error - new_error:+.3f} |")
    source_rows = "\n".join(
        f"| {name} | {values['images']} | {values['old_mean_image_error_px']:.3f} | {values['new_mean_image_error_px']:.3f} | {values['mean_image_improvement_px']:+.3f} | {values['improved_images']}/{values['worsened_images']} |"
        for name, values in manifest["source_analysis"].items()
    )
    rare_rows = "\n".join(
        f"| {name} | {values['truth_vertebrae']} | {values['predicted_vertebrae']} | {values['detected_vertebrae']} | {values['false_positive_vertebrae']} | {format_percent(values['precision'])} | {format_percent(values['point_recall'])} | {values['mean_error_px']} |"
        for name, values in extra["per_class"].items()
    )
    report = f"""# 最新Corner 20类模型与上一版18类模型对比

- test：{manifest['test']['images']}张原图；共同能力只评V0–V17
- 旧模型：`{manifest['models']['old']['path']}`（18类）
- 新模型：`{manifest['models']['new']['path']}`（20类）
- 推理：imgsz={manifest['configuration']['imgsz']}，conf={manifest['configuration']['confidence']}，native class ID，CPU逐图预热后计时
- 公平性：V18/L6与V19/T13不进入旧新共同指标；它们只作为新模型新增能力单列

## 结论

{manifest['automatic_interpretation']}

## V0–V17共同能力

| 指标 | 旧18类 | 新20类 | 新模型变化 |
|---|---:|---:|---:|
| 平均误差(px，已检出点) | {old['mean_error_px']} | {new['mean_error_px']} | {comparison['pooled_mean_error_improvement_px']:+.3f} |
| 中位误差(px) | {old['median_error_px']} | {new['median_error_px']} | {old['median_error_px'] - new['median_error_px']:+.3f} |
| P90误差(px) | {old['p90_error_px']} | {new['p90_error_px']} | {old['p90_error_px'] - new['p90_error_px']:+.3f} |
| 点召回 | {old['point_recall']:.2%} | {new['point_recall']:.2%} | {new['point_recall'] - old['point_recall']:+.2%} |
| PCK@10（漏点计失败） | {old['pck_10_all']:.2%} | {new['pck_10_all']:.2%} | {new['pck_10_all'] - old['pck_10_all']:+.2%} |
| PCK@20（漏点计失败） | {old['pck_20_all']:.2%} | {new['pck_20_all']:.2%} | {new['pck_20_all'] - old['pck_20_all']:+.2%} |
| 完整检出V0–V17图像 | {old['complete_ground_truth_images']}/{manifest['test']['images']} | {new['complete_ground_truth_images']}/{manifest['test']['images']} | {new['complete_ground_truth_images'] - old['complete_ground_truth_images']:+d} |
| 单图模型耗时均值(ms) | {old['timing']['mean_ms']} | {new['timing']['mean_ms']} | {comparison['timing_ratio_new_over_old']:.2f}× |

逐图基础误差：新模型改善{comparison['improved_images']}张、恶化{comparison['worsened_images']}张、近似持平{comparison['near_tie_images']}张；单图平均改善的95% bootstrap CI为{comparison['mean_image_improvement_95ci_px']} px。

## 新增V18/V19能力

test中只有{extra['images']}张图、{extra['truth_vertebrae']}个额外椎体、{extra['visible_points']}个角点，样本很少，因此这里只验证“是否学会”，不能据此判断稳定泛化。

| 类别 | GT椎体 | 预测椎体 | 正确检出 | 假阳性 | 精确率 | 点召回 | 已检出点平均误差(px) |
|---|---:|---:|---:|---:|---:|---:|---:|
{rare_rows}

合计：新模型正确检出{extra['detected_vertebrae']}/{extra['truth_vertebrae']}个额外椎体，但总共输出{extra['predicted_vertebrae']}个额外类别目标，其中假阳性{extra['false_positive_vertebrae']}个、涉及{extra['false_positive_images']}张图，额外类别精确率{format_percent(extra['precision'])}；角点召回{format_percent(extra['point_recall'])}，已检出点平均误差{extra['mean_error_px']} px。旧模型输出头只有18类，结构上不支持V18/V19，不把它记成旧模型性能下降。

## 按来源

| 来源 | 图像 | 旧误差 | 新误差 | 新模型改善 | 改善/恶化图像 |
|---|---:|---:|---:|---:|---:|
{source_rows}

## 按V0–V17类别

| 类别 | 旧误差 | 新误差 | 新模型改善 |
|---|---:|---:|---:|
{chr(10).join(vertebra_rows)}

`representatives/`包含改善最大、恶化最大、中位附近及全部V18/V19 test病例的三栏图；绿色为GT、洋红为旧模型、青色为新模型。
"""
    path.write_text(report, encoding="utf-8")


def build_comparison(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    old_model = YOLO(str(args.old_model))
    new_model = YOLO(str(args.new_model))
    if old_model.task != "pose" or len(old_model.names) != 18:
        raise ValueError(f"old checkpoint must be 18-class Pose, found task={old_model.task}, classes={len(old_model.names)}")
    if new_model.task != "pose" or len(new_model.names) != 20:
        raise ValueError(f"new checkpoint must be 20-class Pose, found task={new_model.task}, classes={len(new_model.names)}")
    pairs = corner_eval.find_pairs(args.image_dir, args.label_dir, args.limit)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    old_predictions, old_timing, old_timing_by_file = corner_eval.predict_dataset(
        args.old_model, pairs, args.imgsz, args.device, args.raw_conf
    )
    new_predictions, new_timing, new_timing_by_file = corner_eval.predict_dataset(
        args.new_model, pairs, args.imgsz, args.device, args.raw_conf
    )

    samples = []
    truth_by_file = {}
    old_by_file = {}
    new_by_file = {}
    rare_samples = []
    for image_path, label_path in pairs:
        with Image.open(image_path) as image:
            image.load()
            width, height = ImageOps.exif_transpose(image).size
        truth = corner_eval.parse_corner_label(label_path, width, height)
        base_truth = filter_classes(truth, BASE_CLASS_IDS)
        extra_truth = filter_classes(truth, EXTRA_CLASS_IDS)
        old_assigned = corner_eval.native_assignments(old_predictions[image_path.name], args.confidence)
        new_assigned = corner_eval.native_assignments(new_predictions[image_path.name], args.confidence)
        old_base = corner_eval.evaluate_assignments(base_truth, filter_classes(old_assigned, BASE_CLASS_IDS), width, height)
        new_base = corner_eval.evaluate_assignments(base_truth, filter_classes(new_assigned, BASE_CLASS_IDS), width, height)
        new_full = corner_eval.evaluate_assignments(truth, new_assigned, width, height)
        new_extra = None
        if extra_truth:
            new_extra = corner_eval.evaluate_assignments(extra_truth, filter_classes(new_assigned, EXTRA_CLASS_IDS), width, height)
        improvement = old_base["mean_error_px"] - new_base["mean_error_px"]
        sample = {
            "filename": image_path.name,
            "image_path": str(image_path.resolve()),
            "image_sha256": corner_eval.sha256_file(image_path),
            "label_sha256": corner_eval.sha256_file(label_path),
            "extra_truth_classes": sorted(extra_truth),
            "new_extra_predicted_classes": sorted(set(new_assigned) & EXTRA_CLASS_IDS),
            "old_base": old_base,
            "new_base": new_base,
            "new_extra": new_extra,
            "new_full": new_full,
            "base_improvement_px": round(improvement, 4),
            "old_ms": old_timing_by_file[image_path.name],
            "new_ms": new_timing_by_file[image_path.name],
        }
        samples.append(sample)
        if new_extra is not None:
            rare_samples.append(sample)
        truth_by_file[image_path.name] = truth
        old_by_file[image_path.name] = filter_classes(old_assigned, BASE_CLASS_IDS)
        new_by_file[image_path.name] = new_assigned

    old_summary = aggregate(samples, "old_base")
    new_summary = aggregate(samples, "new_base")
    old_summary["timing"] = old_timing
    new_summary["timing"] = new_timing
    extra_summary = summarize_extra_predictions(samples)
    deltas = [sample["base_improvement_px"] for sample in samples]
    comparison = {
        "pooled_mean_error_improvement_px": round(old_summary["mean_error_px"] - new_summary["mean_error_px"], 4),
        "pooled_relative_error_improvement": round((old_summary["mean_error_px"] - new_summary["mean_error_px"]) / old_summary["mean_error_px"], 6),
        "mean_image_improvement_px": round(statistics.fmean(deltas), 4),
        "mean_image_improvement_95ci_px": bootstrap_mean_ci(deltas),
        "improved_images": sum(value > 0.05 for value in deltas),
        "worsened_images": sum(value < -0.05 for value in deltas),
        "near_tie_images": sum(abs(value) <= 0.05 for value in deltas),
        "timing_ratio_new_over_old": round(new_timing["mean_ms"] / old_timing["mean_ms"], 4),
    }

    representatives = []
    for index, selected in enumerate(select_representatives(samples), 1):
        image_path = Path(selected["image_path"])
        relative = f"representatives/{index:02d}_{selected['group']}_{image_path.stem}.jpg"
        render_preview(
            image_path,
            truth_by_file[selected["filename"]],
            old_by_file[selected["filename"]],
            new_by_file[selected["filename"]],
            selected["old_base"]["mean_error_px"],
            selected["new_base"]["mean_error_px"],
            args.output_dir / relative,
        )
        representatives.append({
            "filename": selected["filename"],
            "group": selected["group"],
            "base_improvement_px": selected["base_improvement_px"],
            "extra_truth_classes": selected["extra_truth_classes"],
            "preview": relative,
        })

    source_analysis = aggregate_sources(samples)
    interpretation = build_interpretation(old_summary, new_summary, comparison, source_analysis, extra_summary)
    manifest = {
        "schema_version": 1,
        "models": {
            "old": {"path": str(args.old_model.resolve()), "sha256": corner_eval.sha256_file(args.old_model), "class_count": 18},
            "new": {"path": str(args.new_model.resolve()), "sha256": corner_eval.sha256_file(args.new_model), "class_count": 20},
        },
        "configuration": {"imgsz": args.imgsz, "device": args.device, "raw_conf": args.raw_conf, "confidence": args.confidence, "primary_mode": "native"},
        "test": {"image_dir": str(args.image_dir.resolve()), "label_dir": str(args.label_dir.resolve()), "images": len(samples), "extra_images": len(rare_samples)},
        "base_summary": {"old_18class": old_summary, "new_20class": new_summary},
        "base_comparison": comparison,
        "extra_summary": extra_summary,
        "new_full_20class_summary": aggregate(samples, "new_full"),
        "source_analysis": source_analysis,
        "automatic_interpretation": interpretation,
        "representatives": representatives,
        "samples": samples,
    }
    write_csv(args.output_dir / "per_image.csv", samples)
    write_report(args.output_dir / "report.md", manifest)
    manifest["package_files"] = {
        str(path.relative_to(args.output_dir)): corner_eval.sha256_file(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/pose_corner_data/images/test"))
    parser.add_argument("--label-dir", type=Path, default=Path("datasets/pose_corner_data/labels/test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-conf", type=float, default=0.001)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    manifest = build_comparison(parse_args(argv))
    print(json.dumps({
        "base_summary": manifest["base_summary"],
        "base_comparison": manifest["base_comparison"],
        "extra_summary": manifest["extra_summary"],
        "automatic_interpretation": manifest["automatic_interpretation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
