#!/usr/bin/env python3
"""Evaluate one 18-class Corner Pose checkpoint at two inference sizes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps

import compare_corner_models as corner_eval


BASE_CLASS_IDS = frozenset(range(18))


def filter_base_truth(truth: dict[int, corner_eval.CornerObject]) -> dict[int, corner_eval.CornerObject]:
    return {class_id: item for class_id, item in truth.items() if class_id in BASE_CLASS_IDS}


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def bbox_area_fraction(item: corner_eval.CornerObject, width: int, height: int) -> float:
    left, top, right, bottom = item.box_xyxy
    return max(0.0, right - left) * max(0.0, bottom - top) / (width * height)


def size_bin(area: float, lower: float, upper: float) -> str:
    if area <= lower:
        return "small"
    if area <= upper:
        return "medium"
    return "large"


def aggregate_size_bins(samples: Sequence[dict], result_key: str, lower: float, upper: float) -> dict:
    grouped: dict[str, list[dict]] = {name: [] for name in ("small", "medium", "large")}
    for sample in samples:
        areas = {int(class_id): area for class_id, area in sample["truth_area_fraction"].items()}
        for row in sample[result_key]["point_rows"]:
            if not row["visible"]:
                continue
            grouped[size_bin(areas[row["class_id"]], lower, upper)].append(row)
    result = {}
    for name, rows in grouped.items():
        errors = [row["distance_px"] for row in rows if row["distance_px"] is not None]
        result[name] = {
            "visible_points": len(rows),
            "detected_points": len(errors),
            "point_recall": len(errors) / len(rows) if rows else None,
            "mean_error_px": statistics.fmean(errors) if errors else None,
            "p90_error_px": percentile(errors, 0.9),
            "pck20_all": sum(value <= 20 for value in errors) / len(rows) if rows else None,
        }
    return result


def aggregate_sources(samples: Sequence[dict], result_key: str) -> dict:
    output = {}
    for group in sorted({corner_eval.source_group(sample["filename"]) for sample in samples}):
        rows = [sample for sample in samples if corner_eval.source_group(sample["filename"]) == group]
        errors = [sample[result_key]["mean_error_px"] for sample in rows if sample[result_key]["mean_error_px"] is not None]
        output[group] = {
            "sample_count": len(rows),
            "mean_image_error_px": statistics.fmean(errors) if errors else None,
        }
    return output


def render_preview(
    image_path: Path,
    truth: dict[int, corner_eval.CornerObject],
    prediction_800: dict[int, corner_eval.CornerObject],
    prediction_1024: dict[int, corner_eval.CornerObject],
    error_800: float | None,
    error_1024: float | None,
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        source.load()
        source = ImageOps.exif_transpose(source).convert("RGB")
        base, scale = corner_eval.fit_image(source)
    gt_panel = corner_eval.add_title(
        corner_eval.draw_objects(base, truth.values(), corner_eval.GT_COLOR, scale),
        f"GT V0-V17: {len(truth)} vertebrae",
        corner_eval.GT_COLOR,
    )
    panel_800 = corner_eval.draw_objects(base, truth.values(), corner_eval.GT_COLOR, scale)
    panel_800 = corner_eval.add_title(
        corner_eval.draw_objects(panel_800, prediction_800.values(), corner_eval.OLD_COLOR, scale),
        f"imgsz=800: {error_800} px",
        corner_eval.OLD_COLOR,
    )
    panel_1024 = corner_eval.draw_objects(base, truth.values(), corner_eval.GT_COLOR, scale)
    panel_1024 = corner_eval.add_title(
        corner_eval.draw_objects(panel_1024, prediction_1024.values(), corner_eval.NEW_COLOR, scale),
        f"imgsz=1024: {error_1024} px",
        corner_eval.NEW_COLOR,
    )
    gap = 8
    canvas = Image.new("RGB", (gt_panel.width * 3 + gap * 2, gt_panel.height), (70, 70, 70))
    canvas.paste(gt_panel, (0, 0))
    canvas.paste(panel_800, (gt_panel.width + gap, 0))
    canvas.paste(panel_1024, ((gt_panel.width + gap) * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=88, optimize=True)


def select_representatives(samples: Sequence[dict], count_each: int = 4) -> list[dict]:
    comparable = sorted(
        (sample for sample in samples if sample["improvement_px"] is not None),
        key=lambda sample: sample["improvement_px"],
    )
    if not comparable:
        return []
    worst = comparable[:count_each]
    best = comparable[-count_each:]
    middle = len(comparable) // 2
    median = comparable[max(0, middle - count_each // 2) : middle + (count_each + 1) // 2]
    selected = []
    seen = set()
    for group, rows in (("worse", worst), ("median", median), ("better", reversed(best))):
        for sample in rows:
            if sample["filename"] in seen:
                continue
            seen.add(sample["filename"])
            selected.append({"group": group, **sample})
    return selected


def write_csv(path: Path, samples: Sequence[dict]) -> None:
    fields = (
        "filename",
        "source",
        "ignored_extra_classes",
        "error_800_px",
        "error_1024_px",
        "improvement_px",
        "recall_800",
        "recall_1024",
        "predicted_800",
        "predicted_1024",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "filename": sample["filename"],
                    "source": corner_eval.source_group(sample["filename"]),
                    "ignored_extra_classes": ",".join(str(value) for value in sample["ignored_extra_classes"]),
                    "error_800_px": sample["size_800"]["mean_error_px"],
                    "error_1024_px": sample["size_1024"]["mean_error_px"],
                    "improvement_px": sample["improvement_px"],
                    "recall_800": sample["size_800"]["detected_points"] / sample["size_800"]["visible_points"],
                    "recall_1024": sample["size_1024"]["detected_points"] / sample["size_1024"]["visible_points"],
                    "predicted_800": sample["size_800"]["predicted_vertebrae"],
                    "predicted_1024": sample["size_1024"]["predicted_vertebrae"],
                }
            )


def write_report(path: Path, manifest: dict) -> None:
    size_800 = manifest["summary"]["size_800"]
    size_1024 = manifest["summary"]["size_1024"]
    comparison = manifest["comparison"]
    size_rows = "\n".join(
        f"| {name} | {values['object_area_fraction_range']} | {values['size_800']['mean_error_px']:.3f} | {values['size_1024']['mean_error_px']:.3f} | {values['improvement_px']:.3f} | {values['size_800']['point_recall']:.2%} | {values['size_1024']['point_recall']:.2%} |"
        for name, values in manifest["size_analysis"].items()
    )
    vertebra_rows = []
    for class_id in range(18):
        name = f"V{class_id}"
        old_value = size_800["per_vertebra"][name]["mean_error_px"]
        new_value = size_1024["per_vertebra"][name]["mean_error_px"]
        vertebra_rows.append(f"| {name} | {old_value} | {new_value} | {old_value - new_value:.3f} |")
    source_rows = "\n".join(
        f"| {name} | {values['sample_count']} | {values['size_800']:.3f} | {values['size_1024']:.3f} | {values['size_800'] - values['size_1024']:.3f} |"
        for name, values in manifest["source_analysis"].items()
    )
    conclusion = comparison["automatic_interpretation"]
    path.write_text(
        f"""# Corner 18类模型800/1024推理敏感性测试

- 模型：`{manifest['model']['path']}`
- 权重SHA-256：`{manifest['model']['sha256']}`
- test：{manifest['test']['images']}张原图，仅评V0–V17；{manifest['test']['images_with_ignored_extras']}张图的{manifest['test']['ignored_extra_rows']}行V18/V19真值被忽略
- 推理：native class，conf={manifest['configuration']['confidence']}，同一CPU环境预热后逐图计时

## 总体结果

| 指标 | imgsz=800 | imgsz=1024 | 1024变化 |
|---|---:|---:|---:|
| 平均误差(px) | {size_800['mean_error_px']} | {size_1024['mean_error_px']} | {comparison['mean_error_improvement_px']:+.3f} px改善 |
| 中位误差(px) | {size_800['median_error_px']} | {size_1024['median_error_px']} | {size_800['median_error_px'] - size_1024['median_error_px']:+.3f} |
| P90误差(px) | {size_800['p90_error_px']} | {size_1024['p90_error_px']} | {size_800['p90_error_px'] - size_1024['p90_error_px']:+.3f} |
| 点召回 | {size_800['point_recall']:.2%} | {size_1024['point_recall']:.2%} | {size_1024['point_recall'] - size_800['point_recall']:+.2%} |
| PCK@20（漏点计失败） | {size_800['pck_20_all']:.2%} | {size_1024['pck_20_all']:.2%} | {size_1024['pck_20_all'] - size_800['pck_20_all']:+.2%} |
| 单图CPU模型耗时均值 | {size_800['timing']['mean_ms']} ms | {size_1024['timing']['mean_ms']} ms | {comparison['timing_ratio_1024_over_800']:.2f}× |

逐图：1024改善{comparison['improved_images']}张、恶化{comparison['worsened_images']}张、近似持平{comparison['near_tie_images']}张。

## 按椎体框面积三等分

| 尺寸组 | 原图面积占比范围 | 800误差 | 1024误差 | 改善 | 800召回 | 1024召回 |
|---|---|---:|---:|---:|---:|---:|
{size_rows}

## 按来源

| 来源 | 数量 | 800误差 | 1024误差 | 改善 |
|---|---:|---:|---:|---:|
{source_rows}

## 按类别

| 类别 | 800误差 | 1024误差 | 改善 |
|---|---:|---:|---:|
{chr(10).join(vertebra_rows)}

## 自动判读

{conclusion}

`representatives/`保存1024改善最大、恶化最大和中位附近的三栏预览；绿色为GT，洋红为800，青色为1024。
""",
        encoding="utf-8",
    )


def package_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): corner_eval.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def build_evaluation(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    model_check = YOLO(str(args.model))
    if model_check.task != "pose" or len(model_check.names) != 18:
        raise ValueError(f"expected an 18-class Pose checkpoint, found task={model_check.task} classes={len(model_check.names)}")
    pairs = corner_eval.find_pairs(args.image_dir, args.label_dir, args.limit)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    predictions_800, timing_800, timing_by_file_800 = corner_eval.predict_dataset(
        args.model, pairs, 800, args.device, args.raw_conf
    )
    predictions_1024, timing_1024, timing_by_file_1024 = corner_eval.predict_dataset(
        args.model, pairs, 1024, args.device, args.raw_conf
    )

    samples = []
    truth_objects = []
    ignored_extra_rows = 0
    images_with_ignored_extras = 0
    truth_by_file = {}
    assigned_by_file = {"size_800": {}, "size_1024": {}}
    for image_path, label_path in pairs:
        with Image.open(image_path) as image:
            image.load()
            width, height = ImageOps.exif_transpose(image).size
        full_truth = corner_eval.parse_corner_label(label_path, width, height)
        truth = filter_base_truth(full_truth)
        ignored = sorted(set(full_truth) - set(truth))
        ignored_extra_rows += len(ignored)
        images_with_ignored_extras += bool(ignored)
        assigned_800 = corner_eval.native_assignments(predictions_800[image_path.name], args.confidence)
        assigned_1024 = corner_eval.native_assignments(predictions_1024[image_path.name], args.confidence)
        metrics_800 = corner_eval.evaluate_assignments(truth, assigned_800, width, height)
        metrics_1024 = corner_eval.evaluate_assignments(truth, assigned_1024, width, height)
        improvement = None
        if metrics_800["mean_error_px"] is not None and metrics_1024["mean_error_px"] is not None:
            improvement = metrics_800["mean_error_px"] - metrics_1024["mean_error_px"]
        areas = {class_id: bbox_area_fraction(item, width, height) for class_id, item in truth.items()}
        truth_objects.extend(areas.values())
        truth_by_file[image_path.name] = truth
        assigned_by_file["size_800"][image_path.name] = assigned_800
        assigned_by_file["size_1024"][image_path.name] = assigned_1024
        samples.append(
            {
                "filename": image_path.name,
                "image_path": str(image_path),
                "image_sha256": corner_eval.sha256_file(image_path),
                "label_sha256": corner_eval.sha256_file(label_path),
                "ignored_extra_classes": ignored,
                "truth_area_fraction": {str(key): value for key, value in areas.items()},
                "size_800": metrics_800,
                "size_1024": metrics_1024,
                "timing_ms_800": timing_by_file_800[image_path.name],
                "timing_ms_1024": timing_by_file_1024[image_path.name],
                "improvement_px": improvement,
            }
        )

    summary_samples = [
        {"size_800": {"native": sample["size_800"]}, "size_1024": {"native": sample["size_1024"]}}
        for sample in samples
    ]
    summary_800 = corner_eval.aggregate_mode(summary_samples, "size_800", "native")
    summary_1024 = corner_eval.aggregate_mode(summary_samples, "size_1024", "native")
    summary_800["timing"] = timing_800
    summary_1024["timing"] = timing_1024
    lower = percentile(truth_objects, 1 / 3)
    upper = percentile(truth_objects, 2 / 3)
    assert lower is not None and upper is not None
    sizes_800 = aggregate_size_bins(samples, "size_800", lower, upper)
    sizes_1024 = aggregate_size_bins(samples, "size_1024", lower, upper)
    ranges = {
        "small": f"≤{lower:.6f}",
        "medium": f"({lower:.6f}, {upper:.6f}]",
        "large": f">{upper:.6f}",
    }
    size_analysis = {
        name: {
            "object_area_fraction_range": ranges[name],
            "size_800": sizes_800[name],
            "size_1024": sizes_1024[name],
            "improvement_px": sizes_800[name]["mean_error_px"] - sizes_1024[name]["mean_error_px"],
        }
        for name in ("small", "medium", "large")
    }
    source_800 = aggregate_sources(samples, "size_800")
    source_1024 = aggregate_sources(samples, "size_1024")
    source_analysis = {
        name: {
            "sample_count": source_800[name]["sample_count"],
            "size_800": source_800[name]["mean_image_error_px"],
            "size_1024": source_1024[name]["mean_image_error_px"],
        }
        for name in source_800
    }
    comparable = [sample for sample in samples if sample["improvement_px"] is not None]
    mean_improvement = summary_800["mean_error_px"] - summary_1024["mean_error_px"]
    relative_improvement = mean_improvement / summary_800["mean_error_px"]
    pck_gain = summary_1024["pck_20_all"] - summary_800["pck_20_all"]
    if relative_improvement >= 0.05 and pck_gain >= 0.01:
        interpretation = "1024在平均误差和PCK@20上均达到实质改善，值得进一步做1024训练/微调对照，但仍需结合GPU延迟。"
    elif relative_improvement <= 0.02 and pck_gain <= 0.005:
        interpretation = "1024改善很小，当前模型对推理分辨率不敏感；仅为此重训1024的收益不足。"
    else:
        interpretation = "1024有轻度或不一致改善，尚不足以直接重训；建议先看小目标组和耗时，再决定是否做短程1024微调。"
    comparison = {
        "mean_error_improvement_px": mean_improvement,
        "relative_mean_error_improvement": relative_improvement,
        "pck20_gain": pck_gain,
        "improved_images": sum(sample["improvement_px"] > 0.05 for sample in comparable),
        "worsened_images": sum(sample["improvement_px"] < -0.05 for sample in comparable),
        "near_tie_images": sum(abs(sample["improvement_px"]) <= 0.05 for sample in comparable),
        "timing_ratio_1024_over_800": timing_1024["mean_ms"] / timing_800["mean_ms"],
        "automatic_interpretation": interpretation,
    }

    representative_records = []
    for index, selected in enumerate(select_representatives(samples), 1):
        image_path = Path(selected["image_path"])
        relative = f"representatives/{index:02d}_{selected['group']}_{image_path.stem}.jpg"
        render_preview(
            image_path,
            truth_by_file[selected["filename"]],
            assigned_by_file["size_800"][selected["filename"]],
            assigned_by_file["size_1024"][selected["filename"]],
            selected["size_800"]["mean_error_px"],
            selected["size_1024"]["mean_error_px"],
            args.output_dir / relative,
        )
        representative_records.append(
            {
                "filename": selected["filename"],
                "group": selected["group"],
                "improvement_px": selected["improvement_px"],
                "preview": relative,
            }
        )

    manifest = {
        "schema_version": 1,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": corner_eval.sha256_file(args.model),
            "class_count": 18,
        },
        "configuration": {
            "sizes": [800, 1024],
            "device": args.device,
            "raw_conf": args.raw_conf,
            "confidence": args.confidence,
            "evaluation_classes": list(range(18)),
        },
        "test": {
            "image_dir": str(args.image_dir.resolve()),
            "label_dir": str(args.label_dir.resolve()),
            "images": len(samples),
            "images_with_ignored_extras": images_with_ignored_extras,
            "ignored_extra_rows": ignored_extra_rows,
        },
        "summary": {"size_800": summary_800, "size_1024": summary_1024},
        "comparison": comparison,
        "size_analysis": size_analysis,
        "source_analysis": source_analysis,
        "representatives": representative_records,
        "samples": samples,
    }
    write_csv(args.output_dir / "per_image.csv", samples)
    write_report(args.output_dir / "report.md", manifest)
    manifest["package_files"] = package_hashes(args.output_dir)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/pose_corner_data/images/test"))
    parser.add_argument("--label-dir", type=Path, default=Path("datasets/pose_corner_data/labels/test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-conf", type=float, default=0.001)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = build_evaluation(args)
    print(
        json.dumps(
            {"summary": manifest["summary"], "comparison": manifest["comparison"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
