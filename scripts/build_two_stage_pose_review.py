#!/usr/bin/env python3
"""Build an offline review package for full-image versus ROI-refined six-point Pose inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import cv2
from PIL import Image, ImageDraw, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_six_point_model_review import (  # noqa: E402
    BLACK,
    GT_COLOR,
    KEYPOINT_NAMES,
    PoseLabel,
    add_title,
    choose_font,
    draw_points,
    find_pairs,
    fit_image,
    parse_pose_label,
    sha256_file,
)
from two_stage_pose_inference import PosePrediction, TwoStageResult, result_to_dict, two_stage_predict  # noqa: E402


FIRST_COLOR = (235, 65, 185)
FINAL_COLOR = (45, 195, 240)
ROI_COLOR = (245, 190, 35)
WHITE = (255, 255, 255)


def ensure_numpy_checkpoint_compatibility() -> bool:
    """Alias NumPy 2's pickle module path when loading its checkpoints on NumPy 1.x."""
    try:
        importlib.import_module("numpy._core.multiarray")
        return False
    except ModuleNotFoundError:
        import numpy as np

        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        return True


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def source_group(filename: str) -> str:
    return "eap" if filename.startswith("eap_") else "old"


def normalized_prediction(prediction: PosePrediction) -> tuple[tuple[float, float, float], ...]:
    return prediction.normalized_keypoints()


def calculate_stage_metrics(label: PoseLabel, prediction: PosePrediction, width: int, height: int) -> dict:
    diagonal = math.hypot(width, height)
    points = []
    for name, truth, predicted, confidence in zip(
        KEYPOINT_NAMES,
        label.keypoints,
        prediction.keypoints_xy,
        prediction.keypoint_confidences,
    ):
        gt_x_norm, gt_y_norm, visibility = truth
        pred_x, pred_y = predicted
        detected = confidence > 0 and (pred_x != 0 or pred_y != 0)
        dx = dy = distance = distance_diagonal = None
        if visibility > 0 and detected:
            gt_x, gt_y = gt_x_norm * width, gt_y_norm * height
            dx, dy = pred_x - gt_x, pred_y - gt_y
            distance = math.hypot(dx, dy)
            distance_diagonal = distance / diagonal
        points.append(
            {
                "name": name,
                "visible": visibility > 0,
                "detected": detected,
                "confidence": rounded(confidence, 6),
                "dx_px": rounded(dx),
                "dy_px": rounded(dy),
                "distance_px": rounded(distance),
                "distance_diagonal": rounded(distance_diagonal, 6),
            }
        )
    measured = [point for point in points if point["distance_px"] is not None]
    errors = [point["distance_px"] for point in measured]
    normalized_errors = [point["distance_diagonal"] for point in measured]
    shoulders = [point for point in measured if point["name"] in ("CR", "CL")]
    lower = [point for point in measured if point["name"] not in ("CR", "CL")]

    def mean(field: str, values: Sequence[dict]) -> float | None:
        return None if not values else sum(point[field] for point in values) / len(values)

    span_bias = None
    if len(measured) == 6:
        gt_shoulder_y = statistics.mean(label.keypoints[index][1] * height for index in (0, 1))
        gt_lower_y = statistics.mean(label.keypoints[index][1] * height for index in (2, 3, 4, 5))
        pred_shoulder_y = statistics.mean(prediction.keypoints_xy[index][1] for index in (0, 1))
        pred_lower_y = statistics.mean(prediction.keypoints_xy[index][1] for index in (2, 3, 4, 5))
        span_bias = (pred_lower_y - pred_shoulder_y) - (gt_lower_y - gt_shoulder_y)
    return {
        "points": points,
        "visible_count": sum(point["visible"] for point in points),
        "detected_visible_count": len(measured),
        "missing_count": sum(point["visible"] for point in points) - len(measured),
        "mean_error_px": rounded(None if not errors else statistics.mean(errors)),
        "median_error_px": rounded(percentile(errors, 0.5)),
        "max_error_px": rounded(None if not errors else max(errors)),
        "mean_error_diagonal": rounded(None if not normalized_errors else statistics.mean(normalized_errors), 6),
        "pck_20px_hits": sum(value <= 20 for value in errors),
        "pck_2pct_hits": sum(value <= 0.02 for value in normalized_errors),
        "mean_dy_px": rounded(mean("dy_px", measured)),
        "shoulder_mean_dy_px": rounded(mean("dy_px", shoulders)),
        "lower_mean_dy_px": rounded(mean("dy_px", lower)),
        "span_bias_px": rounded(span_bias),
    }


def aggregate_stage(samples: Sequence[dict], metric_key: str) -> dict:
    metrics = [sample[metric_key] for sample in samples]
    points = [point for metric in metrics for point in metric["points"] if point["visible"]]
    measured = [point for point in points if point["distance_px"] is not None]
    errors = [point["distance_px"] for point in measured]
    diagonal_errors = [point["distance_diagonal"] for point in measured]

    def mean_values(values: Sequence[float]) -> float | None:
        return None if not values else statistics.mean(values)

    per_keypoint = {}
    for name in KEYPOINT_NAMES:
        named = [point for point in points if point["name"] == name]
        valid = [point for point in named if point["distance_px"] is not None]
        per_keypoint[name] = {
            "detected": len(valid),
            "total": len(named),
            "mean_error_px": rounded(mean_values([point["distance_px"] for point in valid])),
            "mean_dx_px": rounded(mean_values([point["dx_px"] for point in valid])),
            "mean_dy_px": rounded(mean_values([point["dy_px"] for point in valid])),
            "above_truth_fraction": rounded(None if not valid else sum(point["dy_px"] < 0 for point in valid) / len(valid), 6),
            "pck_20px": rounded(None if not valid else sum(point["distance_px"] <= 20 for point in valid) / len(valid), 6),
            "pck_2pct_diagonal": rounded(
                None if not valid else sum(point["distance_diagonal"] <= 0.02 for point in valid) / len(valid), 6
            ),
        }
    stage = {
        "sample_count": len(samples),
        "visible_points": len(points),
        "detected_points": len(measured),
        "point_recall": rounded(None if not points else len(measured) / len(points), 6),
        "samples_with_all_six": sum(metric["missing_count"] == 0 for metric in metrics),
        "mean_error_px": rounded(mean_values(errors)),
        "median_error_px": rounded(percentile(errors, 0.5)),
        "p90_error_px": rounded(percentile(errors, 0.9)),
        "mean_error_diagonal": rounded(mean_values(diagonal_errors), 6),
        "pck_20px": rounded(None if not measured else sum(value <= 20 for value in errors) / len(measured), 6),
        "pck_2pct_diagonal": rounded(
            None if not measured else sum(value <= 0.02 for value in diagonal_errors) / len(measured), 6
        ),
        "mean_dy_px": rounded(mean_values([metric["mean_dy_px"] for metric in metrics if metric["mean_dy_px"] is not None])),
        "shoulder_mean_dy_px": rounded(
            mean_values([metric["shoulder_mean_dy_px"] for metric in metrics if metric["shoulder_mean_dy_px"] is not None])
        ),
        "lower_mean_dy_px": rounded(
            mean_values([metric["lower_mean_dy_px"] for metric in metrics if metric["lower_mean_dy_px"] is not None])
        ),
        "mean_span_bias_px": rounded(mean_values([metric["span_bias_px"] for metric in metrics if metric["span_bias_px"] is not None])),
        "per_keypoint": per_keypoint,
    }
    return stage


def distribution(values: Sequence[float]) -> dict:
    return {
        "count": len(values),
        "mean_ms": rounded(None if not values else statistics.mean(values)),
        "median_ms": rounded(percentile(values, 0.5)),
        "p90_ms": rounded(percentile(values, 0.9)),
        "min_ms": rounded(None if not values else min(values)),
        "max_ms": rounded(None if not values else max(values)),
    }


def aggregate_summary(samples: Sequence[dict], evaluation_wall_seconds: float) -> dict:
    improvements = [
        sample["first_metrics"]["mean_error_px"] - sample["final_metrics"]["mean_error_px"]
        for sample in samples
        if sample["first_metrics"]["mean_error_px"] is not None and sample["final_metrics"]["mean_error_px"] is not None
    ]
    first_times = [sample["timing_ms"]["first_inference"] for sample in samples]
    second_times = [sample["timing_ms"]["second_inference"] for sample in samples if sample["timing_ms"]["second_inference"] is not None]
    total_times = [sample["timing_ms"]["total_inference"] for sample in samples]
    end_times = [sample["timing_ms"]["end_to_end"] for sample in samples]
    fallbacks = {}
    for sample in samples:
        if sample["fallback_reason"]:
            fallbacks[sample["fallback_reason"]] = fallbacks.get(sample["fallback_reason"], 0) + 1
    by_source = {}
    for source in ("eap", "old"):
        group = [sample for sample in samples if sample["source"] == source]
        group_improvements = [
            sample["first_metrics"]["mean_error_px"] - sample["final_metrics"]["mean_error_px"]
            for sample in group
            if sample["first_metrics"]["mean_error_px"] is not None and sample["final_metrics"]["mean_error_px"] is not None
        ]
        by_source[source] = {
            "sample_count": len(group),
            "first": aggregate_stage(group, "first_metrics"),
            "final": aggregate_stage(group, "final_metrics"),
            "mean_improvement_px": rounded(None if not group_improvements else statistics.mean(group_improvements)),
            "improved_sample_count": sum(value > 0 for value in group_improvements),
        }
    return {
        "sample_count": len(samples),
        "second_stage_used": sum(sample["used_second_stage"] for sample in samples),
        "fallbacks": fallbacks,
        "first": aggregate_stage(samples, "first_metrics"),
        "final": aggregate_stage(samples, "final_metrics"),
        "comparison": {
            "mean_improvement_px": rounded(None if not improvements else statistics.mean(improvements)),
            "median_improvement_px": rounded(percentile(improvements, 0.5)),
            "improved_sample_count": sum(value > 0 for value in improvements),
            "unchanged_sample_count": sum(value == 0 for value in improvements),
            "worsened_sample_count": sum(value < 0 for value in improvements),
        },
        "timing": {
            "definition": "model timings exclude disk decode and preview rendering; end_to_end includes one image decode and inference",
            "first_inference": distribution(first_times),
            "second_inference": distribution(second_times),
            "total_inference": distribution(total_times),
            "end_to_end": distribution(end_times),
            "sequential_inference_fps": rounded(None if not total_times else 1000.0 / statistics.mean(total_times), 3),
            "evaluation_wall_seconds": rounded(evaluation_wall_seconds, 3),
        },
        "by_source": by_source,
    }


def draw_error_lines(
    image: Image.Image,
    truth: Sequence[tuple[float, float, int]],
    predicted: Sequence[tuple[float, float, float]],
    color: tuple[int, int, int],
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width = max(1, round(min(output.size) * 0.0025))
    for gt, pred in zip(truth, predicted):
        if gt[2] <= 0 or pred[2] <= 0:
            continue
        draw.line(
            (gt[0] * output.width, gt[1] * output.height, pred[0] * output.width, pred[1] * output.height),
            fill=color,
            width=width,
        )
    return output


def draw_roi(image: Image.Image, roi, original_width: int, original_height: int) -> Image.Image:
    if roi is None:
        return image
    output = image.copy()
    x1, y1, x2, y2 = roi
    scaled = (
        round(x1 / original_width * output.width),
        round(y1 / original_height * output.height),
        round(x2 / original_width * output.width),
        round(y2 / original_height * output.height),
    )
    ImageDraw.Draw(output).rectangle(scaled, outline=ROI_COLOR, width=max(2, round(min(output.size) * 0.004)))
    return output


def stage_panel(
    base: Image.Image,
    label: PoseLabel,
    prediction: PosePrediction,
    color: tuple[int, int, int],
    prefix: str,
    title: str,
    roi,
    width: int,
    height: int,
) -> Image.Image:
    predicted = normalized_prediction(prediction)
    panel = draw_roi(base, roi, width, height)
    panel = draw_error_lines(panel, label.keypoints, predicted, color)
    panel = draw_points(panel, label.keypoints, GT_COLOR, "GT-")
    panel = draw_points(panel, predicted, color, prefix)
    return add_title(panel, title, WHITE)


def render_preview(
    image_path: Path,
    label: PoseLabel,
    result: TwoStageResult,
    first_metrics: dict,
    final_metrics: dict,
    output_path: Path,
    max_panel_width: int = 800,
    max_panel_height: int = 1100,
) -> tuple[int, int]:
    with Image.open(image_path) as source:
        source.load()
        source = ImageOps.exif_transpose(source).convert("RGB")
        original_size = source.size
        base, _ = fit_image(source, max_panel_width, max_panel_height)
    truth = add_title(draw_points(base, label.keypoints, GT_COLOR, "GT-"), "Ground truth", GT_COLOR)
    first_title = f"Stage 1  err={first_metrics['mean_error_px']}px  {result.first_inference_ms:.1f}ms"
    final_status = "ROI refined" if result.used_second_stage else f"Fallback: {result.fallback_reason}"
    final_title = f"Stage 2  err={final_metrics['mean_error_px']}px  {final_status}"
    first = stage_panel(base, label, result.first, FIRST_COLOR, "P1-", first_title, result.roi_xyxy, *original_size)
    final = stage_panel(base, label, result.final, FINAL_COLOR, "P2-", final_title, result.roi_xyxy, *original_size)
    gap = 8
    canvas = Image.new("RGB", (truth.width * 3 + gap * 2, truth.height), (70, 70, 70))
    canvas.paste(truth, (0, 0))
    canvas.paste(first, (truth.width + gap, 0))
    canvas.paste(final, ((truth.width + gap) * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=90, optimize=True)
    return original_size


def write_csv(path: Path, samples: Sequence[dict]) -> None:
    fields = (
        "rank",
        "filename",
        "source",
        "used_second_stage",
        "fallback_reason",
        "first_mean_error_px",
        "final_mean_error_px",
        "improvement_px",
        "first_shoulder_dy_px",
        "final_shoulder_dy_px",
        "first_lower_dy_px",
        "final_lower_dy_px",
        "first_span_bias_px",
        "final_span_bias_px",
        "first_inference_ms",
        "second_inference_ms",
        "total_inference_ms",
        "end_to_end_ms",
        "preview",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, sample in enumerate(samples, 1):
            first, final, timing = sample["first_metrics"], sample["final_metrics"], sample["timing_ms"]
            writer.writerow(
                {
                    "rank": rank,
                    "filename": sample["filename"],
                    "source": sample["source"],
                    "used_second_stage": sample["used_second_stage"],
                    "fallback_reason": sample["fallback_reason"] or "",
                    "first_mean_error_px": first["mean_error_px"],
                    "final_mean_error_px": final["mean_error_px"],
                    "improvement_px": sample["improvement_px"],
                    "first_shoulder_dy_px": first["shoulder_mean_dy_px"],
                    "final_shoulder_dy_px": final["shoulder_mean_dy_px"],
                    "first_lower_dy_px": first["lower_mean_dy_px"],
                    "final_lower_dy_px": final["lower_mean_dy_px"],
                    "first_span_bias_px": first["span_bias_px"],
                    "final_span_bias_px": final["span_bias_px"],
                    "first_inference_ms": timing["first_inference"],
                    "second_inference_ms": timing["second_inference"],
                    "total_inference_ms": timing["total_inference"],
                    "end_to_end_ms": timing["end_to_end"],
                    "preview": sample["preview"],
                }
            )


def stage_report_table(first: dict, final: dict) -> str:
    rows = (
        ("点召回", f"{first['detected_points']}/{first['visible_points']}", f"{final['detected_points']}/{final['visible_points']}"),
        ("平均误差", f"{first['mean_error_px']} px", f"{final['mean_error_px']} px"),
        ("中位误差", f"{first['median_error_px']} px", f"{final['median_error_px']} px"),
        ("P90误差", f"{first['p90_error_px']} px", f"{final['p90_error_px']} px"),
        ("PCK@20px", f"{first['pck_20px']:.2%}", f"{final['pck_20px']:.2%}"),
        ("PCK@2%对角线", f"{first['pck_2pct_diagonal']:.2%}", f"{final['pck_2pct_diagonal']:.2%}"),
        ("肩点平均dy", f"{first['shoulder_mean_dy_px']} px", f"{final['shoulder_mean_dy_px']} px"),
        ("下四点平均dy", f"{first['lower_mean_dy_px']} px", f"{final['lower_mean_dy_px']} px"),
        ("纵向跨度偏差", f"{first['mean_span_bias_px']} px", f"{final['mean_span_bias_px']} px"),
    )
    return "\n".join(["| 指标 | 首轮原图 | 二阶段结果 |", "|---|---:|---:|", *(f"| {a} | {b} | {c} |" for a, b, c in rows)])


def write_reports(output_dir: Path, model_path: Path, summary: dict, configuration: dict) -> None:
    first, final, comparison, timing = summary["first"], summary["final"], summary["comparison"], summary["timing"]
    table = stage_report_table(first, final)
    readme = f"""# 六点Pose两阶段模型评测包

双击`打开两阶段评测页面.html`查看175张逐图结果。每张预览从左到右为人工真值、首轮原图预测叠加、ROI二阶段预测叠加；绿色为真值，洋红为首轮，青色为二阶段，黄色框为ROI。

- 模型：`{model_path}`
- 原始test：{summary['sample_count']}张
- 二阶段实际使用：{summary['second_stage_used']}/{summary['sample_count']}
- fallback：{summary['fallbacks']}
- 二阶段平均误差改善：{comparison['mean_improvement_px']} px
- 改善/不变/恶化：{comparison['improved_sample_count']}/{comparison['unchanged_sample_count']}/{comparison['worsened_sample_count']}
- 平均首轮推理：{timing['first_inference']['mean_ms']} ms
- 平均次轮推理：{timing['second_inference']['mean_ms']} ms
- 平均两阶段链路：{timing['total_inference']['mean_ms']} ms
- 顺序吞吐：{timing['sequential_inference_fps']} FPS

{table}

`分析报告.md`包含分点、来源和耗时明细；`逐图指标.csv`可用于排序；`manifest.json`保存模型/输入/预览哈希、完整坐标和指标。计时已经过{configuration['warmup']}次真实图预热，模型时间不含磁盘读取与预览绘制，端到端列包含一次图像解码。
"""
    per_point_rows = []
    for name in KEYPOINT_NAMES:
        f, s = first["per_keypoint"][name], final["per_keypoint"][name]
        per_point_rows.append(
            f"| {name} | {f['mean_error_px']} | {s['mean_error_px']} | {f['mean_dy_px']} | {s['mean_dy_px']} | {f['above_truth_fraction']:.2%} | {s['above_truth_fraction']:.2%} |"
        )
    source_rows = []
    for name, values in summary["by_source"].items():
        source_rows.append(
            f"| {name} | {values['sample_count']} | {values['first']['mean_error_px']} | {values['final']['mean_error_px']} | {values['mean_improvement_px']} | {values['improved_sample_count']} |"
        )
    report = f"""# 两阶段六点Pose自动分析报告

## 总体对比

{table}

正数`dy`表示预测点低于标注，负数表示预测点高于标注。纵向跨度偏差为“预测下四点到肩点的平均跨度 - 标注跨度”，正数表示预测上下跨度扩大。

## 分点对比

| 点 | 首轮误差px | 最终误差px | 首轮dy px | 最终dy px | 首轮高于标注比例 | 最终高于标注比例 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_point_rows)}

## 来源对比

| 来源 | 样本 | 首轮误差px | 最终误差px | 平均改善px | 改善样本数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(source_rows)}

## 推理耗时

| 阶段 | 平均ms | 中位ms | P90 ms | 最小ms | 最大ms |
|---|---:|---:|---:|---:|---:|
| 首轮原图 | {timing['first_inference']['mean_ms']} | {timing['first_inference']['median_ms']} | {timing['first_inference']['p90_ms']} | {timing['first_inference']['min_ms']} | {timing['first_inference']['max_ms']} |
| 次轮ROI | {timing['second_inference']['mean_ms']} | {timing['second_inference']['median_ms']} | {timing['second_inference']['p90_ms']} | {timing['second_inference']['min_ms']} | {timing['second_inference']['max_ms']} |
| 两阶段链路 | {timing['total_inference']['mean_ms']} | {timing['total_inference']['median_ms']} | {timing['total_inference']['p90_ms']} | {timing['total_inference']['min_ms']} | {timing['total_inference']['max_ms']} |
| 解码+推理 | {timing['end_to_end']['mean_ms']} | {timing['end_to_end']['median_ms']} | {timing['end_to_end']['p90_ms']} | {timing['end_to_end']['min_ms']} | {timing['end_to_end']['max_ms']} |

- 顺序推理吞吐：{timing['sequential_inference_fps']} FPS。
- 全包评测墙钟时间：{timing['evaluation_wall_seconds']}秒，包含预览渲染和文件哈希，不能当作线上接口延迟。
- 运行设备：`{configuration['device'] or 'auto'}`；imgsz={configuration['imgsz']}，conf={configuration['conf']}，ROI margin={configuration['roi_margin']}。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "分析报告.md").write_text(report, encoding="utf-8")


HTML_TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>六点Pose两阶段评测</title><style>
*{box-sizing:border-box}body{margin:0;background:#11151b;color:#edf2f7;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.top{position:sticky;top:0;z-index:3;background:#18202bee;padding:12px 18px;border-bottom:1px solid #34404f}.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button,select,input,textarea{background:#202b38;color:#fff;border:1px solid #46566a;border-radius:7px;padding:7px 9px}button.active{border-color:#63b3ed;background:#174d72}.stats{color:#afc1d8;margin-top:7px}.main{max-width:1800px;margin:auto;padding:15px}.card{background:#1a222d;border:1px solid #34404f;border-radius:10px;padding:14px}.preview{width:100%;display:block;background:#000}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;margin:12px 0}.metric{background:#111821;border-radius:7px;padding:9px}.good{color:#65e097}.bad{color:#ff8080}.warn{color:#ffcc66}.notes{width:100%;min-height:58px;margin-top:9px}.foot{display:flex;justify-content:space-between;margin-top:10px}.hint{color:#95a6bb;font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:5px;border-bottom:1px solid #34404f;text-align:right}td:first-child,th:first-child{text-align:left}</style></head><body>
<div class="top"><div class="row"><b>六点Pose两阶段评测</b><input id="search" placeholder="文件名"><select id="source"><option value="all">全部来源</option><option value="eap">eap新数据</option><option value="old">旧数据</option></select><select id="status"><option value="all">全部状态</option><option value="used">使用二阶段</option><option value="fallback">fallback</option><option value="improved">二阶段改善</option><option value="worsened">二阶段恶化</option><option value="todo">未人工核验</option></select><select id="sort"><option value="final_desc">最终误差从大到小</option><option value="improvement_asc">恶化最多优先</option><option value="improvement_desc">改善最多优先</option><option value="time_desc">耗时从大到小</option><option value="filename">文件名</option></select><button id="prev">←</button><button id="next">→</button><input id="jump" type="number" min="1" style="width:80px"><button id="export">导出人工结果</button></div><div class="stats" id="stats"></div></div>
<main class="main"><section class="card"><h2 id="title"></h2><img class="preview" id="preview"><div class="meta" id="meta"></div><div id="pointTable"></div><div class="row"><b>人工判断：</b><button data-result="accurate">1 准确</button><button data-result="inaccurate">2 不准确</button><button data-result="unsure">3 不确定</button><button data-result="">清除</button></div><textarea id="notes" class="notes" placeholder="记录偏差位置、裁剪问题或其他观察"></textarea><div class="foot"><span class="hint">快捷键：←/→切换，1准确，2不准确，3不确定；结果自动保存在浏览器。</span><span id="position"></span></div></section></main>
<script src="review_data.js"></script><script>(()=>{'use strict';const pkg=window.REVIEW_PACKAGE,samples=pkg.samples,key='two-stage-pose-review-'+pkg.package_id;let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}let filtered=[],pos=0;const $=id=>document.getElementById(id);const save=()=>localStorage.setItem(key,JSON.stringify(state));
function refresh(){const q=$('search').value.toLowerCase(),src=$('source').value,status=$('status').value,sort=$('sort').value;filtered=samples.map((s,i)=>i).filter(i=>{const s=samples[i],human=(state[s.filename]||{}).result||'';if(q&&!s.filename.toLowerCase().includes(q))return false;if(src!=='all'&&s.source!==src)return false;if(status==='used'&&!s.used_second_stage)return false;if(status==='fallback'&&s.used_second_stage)return false;if(status==='improved'&&!(s.improvement_px>0))return false;if(status==='worsened'&&!(s.improvement_px<0))return false;if(status==='todo'&&human)return false;return true});filtered.sort((a,b)=>{const x=samples[a],y=samples[b];if(sort==='improvement_asc')return x.improvement_px-y.improvement_px;if(sort==='improvement_desc')return y.improvement_px-x.improvement_px;if(sort==='time_desc')return y.timing_ms.total_inference-x.timing_ms.total_inference;if(sort==='filename')return x.filename.localeCompare(y.filename);return (y.final_metrics.mean_error_px??-1)-(x.final_metrics.mean_error_px??-1)});pos=Math.min(pos,Math.max(0,filtered.length-1));render()}
function render(){const reviewed=Object.values(state).filter(v=>v.result).length,sm=pkg.summary;$('stats').textContent=`175张自动汇总：首轮 ${sm.first.mean_error_px}px → 最终 ${sm.final.mean_error_px}px；平均改善 ${sm.comparison.mean_improvement_px}px；两阶段平均 ${sm.timing.total_inference.mean_ms}ms；人工已核验 ${reviewed}/${samples.length}`;if(!filtered.length){$('title').textContent='当前筛选无样本';$('preview').removeAttribute('src');$('meta').innerHTML='';$('pointTable').innerHTML='';$('position').textContent='0/0';return}const s=samples[filtered[pos]],f=s.first_metrics,r=s.final_metrics,t=s.timing_ms,e=state[s.filename]||{},cls=s.improvement_px>=0?'good':'bad';$('title').textContent=`${s.filename}`;$('preview').src=s.preview;$('meta').innerHTML=`<div class="metric">来源：<b>${s.source}</b></div><div class="metric">首轮误差：<b>${f.mean_error_px}px</b></div><div class="metric">最终误差：<b>${r.mean_error_px}px</b></div><div class="metric">改善：<b class="${cls}">${s.improvement_px}px</b></div><div class="metric">状态：<b>${s.used_second_stage?'ROI二阶段':'fallback '+s.fallback_reason}</b></div><div class="metric">耗时：<b>${t.first_inference}/${t.second_inference??'-'}/${t.total_inference}ms</b></div><div class="metric">肩dy：<b>${f.shoulder_mean_dy_px} → ${r.shoulder_mean_dy_px}px</b></div><div class="metric">下四点dy：<b>${f.lower_mean_dy_px} → ${r.lower_mean_dy_px}px</b></div>`;$('pointTable').innerHTML='<table><tr><th>点</th><th>首轮误差</th><th>最终误差</th><th>首轮dy</th><th>最终dy</th></tr>'+f.points.map((p,i)=>`<tr><td>${p.name}</td><td>${p.distance_px??'-'}</td><td>${r.points[i].distance_px??'-'}</td><td>${p.dy_px??'-'}</td><td>${r.points[i].dy_px??'-'}</td></tr>`).join('')+'</table>';$('notes').value=e.notes||'';document.querySelectorAll('[data-result]').forEach(b=>b.classList.toggle('active',b.dataset.result===(e.result||'')));$('position').textContent=`${pos+1}/${filtered.length}`;$('jump').value=pos+1}
function setResult(result){if(!filtered.length)return;const s=samples[filtered[pos]];state[s.filename]={...(state[s.filename]||{}),result,notes:$('notes').value};save();if(result&&pos<filtered.length-1)pos++;render()}function move(d){if(!filtered.length)return;pos=Math.max(0,Math.min(filtered.length-1,pos+d));render()}document.querySelectorAll('[data-result]').forEach(b=>b.onclick=()=>setResult(b.dataset.result));$('notes').oninput=()=>{if(!filtered.length)return;const s=samples[filtered[pos]];state[s.filename]={...(state[s.filename]||{}),notes:$('notes').value};save()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);['search','source','status','sort'].forEach(id=>$(id).oninput=()=>{pos=0;refresh()});$('jump').onchange=()=>{const i=Number($('jump').value)-1;if(i>=0&&i<filtered.length){pos=i;render()}};$('jump').onkeydown=e=>{if(e.key==='Enter')$('jump').dispatchEvent(new Event('change'))};$('export').onclick=()=>{const rows=['filename,result,notes'];samples.forEach(s=>{const e=state[s.filename]||{},q=v=>'"'+String(v??'').replaceAll('"','""')+'"';rows.push([s.filename,e.result||'',e.notes||''].map(q).join(','))});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+rows.join('\r\n')],{type:'text/csv'}));a.download='两阶段Pose人工核验结果.csv';a.click()};document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(['1','2','3'].includes(e.key))setResult({1:'accurate',2:'inaccurate',3:'unsure'}[e.key])};refresh()})();</script></body></html>'''


def build_package(
    image_dir: Path,
    label_dir: Path,
    model_path: Path,
    output_dir: Path,
    predictor: Callable[[Path], TwoStageResult],
    configuration: dict,
    second_model_path: Path | None = None,
) -> dict:
    pairs = find_pairs(image_dir, label_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".building")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    (staging / "previews").mkdir(parents=True)
    samples = []
    try:
        for _ in range(configuration["warmup"]):
            predictor(pairs[0][0])
        evaluation_start = time.perf_counter()
        for index, (image_path, label_path) in enumerate(pairs, 1):
            label = parse_pose_label(label_path)
            wall_start = time.perf_counter()
            result = predictor(image_path)
            end_to_end_ms = (time.perf_counter() - wall_start) * 1000.0
            width, height = result.first.image_width, result.first.image_height
            first_metrics = calculate_stage_metrics(label, result.first, width, height)
            final_metrics = calculate_stage_metrics(label, result.final, width, height)
            preview_name = f"{index:04d}_{image_path.stem}.jpg"
            preview_path = staging / "previews" / preview_name
            render_preview(image_path, label, result, first_metrics, final_metrics, preview_path)
            timing = {
                "first_inference": rounded(result.first_inference_ms),
                "second_inference": rounded(result.second_inference_ms),
                "total_inference": rounded(result.total_inference_ms),
                "end_to_end": rounded(end_to_end_ms),
            }
            improvement = None
            if first_metrics["mean_error_px"] is not None and final_metrics["mean_error_px"] is not None:
                improvement = rounded(first_metrics["mean_error_px"] - final_metrics["mean_error_px"])
            samples.append(
                {
                    "filename": image_path.name,
                    "source": source_group(image_path.name),
                    "image_sha256": sha256_file(image_path),
                    "label_sha256": sha256_file(label_path),
                    "preview": f"previews/{preview_name}",
                    "preview_sha256": sha256_file(preview_path),
                    "width": width,
                    "height": height,
                    "used_second_stage": result.used_second_stage,
                    "fallback_reason": result.fallback_reason,
                    "roi_xyxy": result.roi_xyxy,
                    "timing_ms": timing,
                    "improvement_px": improvement,
                    "first_prediction": result_to_dict(result)["first"],
                    "final_prediction": result_to_dict(result)["final"],
                    "first_metrics": first_metrics,
                    "final_metrics": final_metrics,
                }
            )
            print(
                f"[{index}/{len(pairs)}] {image_path.name}: "
                f"{first_metrics['mean_error_px']} -> {final_metrics['mean_error_px']} px, "
                f"time={timing['total_inference']} ms fallback={result.fallback_reason}",
                flush=True,
            )
        evaluation_wall_seconds = time.perf_counter() - evaluation_start
        samples.sort(
            key=lambda sample: (
                sample["final_metrics"]["mean_error_px"] is not None,
                sample["final_metrics"]["mean_error_px"] or float("inf"),
            ),
            reverse=True,
        )
        summary = aggregate_summary(samples, evaluation_wall_seconds)
        model_sha = sha256_file(model_path)
        second_model_sha = sha256_file(second_model_path) if second_model_path else model_sha
        args_path = model_path.parents[1] / "args.yaml"
        second_args_path = second_model_path.parents[1] / "args.yaml" if second_model_path else args_path
        package_id = hashlib.sha256(
            (model_sha + "|" + second_model_sha + "|" + "|".join(sample["image_sha256"] for sample in samples) + "|two-stage").encode()
        ).hexdigest()[:16]
        manifest = {
            "package_id": package_id,
            "model": str(model_path.resolve()),
            "model_sha256": model_sha,
            "first_stage_model": str(model_path.resolve()),
            "first_stage_model_sha256": model_sha,
            "second_stage_model": str(second_model_path.resolve()) if second_model_path else str(model_path.resolve()),
            "second_stage_model_sha256": second_model_sha,
            "training_args": str(args_path.resolve()) if args_path.is_file() else None,
            "training_args_sha256": sha256_file(args_path) if args_path.is_file() else None,
            "second_stage_training_args": str(second_args_path.resolve()) if second_args_path.is_file() else None,
            "second_stage_training_args_sha256": sha256_file(second_args_path) if second_args_path.is_file() else None,
            "image_dir": str(image_dir.resolve()),
            "label_dir": str(label_dir.resolve()),
            "configuration": configuration,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "keypoint_names": KEYPOINT_NAMES,
            "summary": summary,
            "samples": samples,
        }
        write_csv(staging / "逐图指标.csv", samples)
        write_reports(staging, model_path, summary, configuration)
        (staging / "打开两阶段评测页面.html").write_text(HTML_TEMPLATE, encoding="utf-8")
        review_data = {"package_id": package_id, "summary": summary, "samples": samples}
        (staging / "review_data.js").write_text(
            "window.REVIEW_PACKAGE=" + json.dumps(review_data, ensure_ascii=False, separators=(",", ":")) + ";\n",
            encoding="utf-8",
        )
        package_files = {}
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            package_files[str(path.relative_to(staging))] = sha256_file(path)
        manifest["package_files"] = package_files
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_bgr(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def make_ultralytics_predictor(
    model_path: Path,
    configuration: dict,
    second_model_path: Path | None = None,
) -> Callable[[Path], TwoStageResult]:
    from ultralytics import YOLO

    # Import Ultralytics/PyTorch before installing the pickle alias. Registering
    # numpy._core before PyTorch initializes can crash older NumPy builds.
    ensure_numpy_checkpoint_compatibility()
    model = YOLO(str(model_path))
    second_model = YOLO(str(second_model_path)) if second_model_path else None

    def predict(image_path: Path) -> TwoStageResult:
        return two_stage_predict(
            model,
            load_bgr(image_path),
            confidence=configuration["conf"],
            image_size=configuration["imgsz"],
            roi_margin=configuration["roi_margin"],
            minimum_first_box_confidence=configuration["roi_conf"],
            device=configuration["device"],
            second_model=second_model,
        )

    return predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--second-model", type=Path, help="optional dedicated ROI refiner; default reuses --model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--roi-margin", type=float, default=0.20)
    parser.add_argument("--roi-conf", type=float, default=0.25)
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "roi_margin": args.roi_margin,
        "roi_conf": args.roi_conf,
        "device": args.device,
        "warmup": args.warmup,
        "second_model": str(args.second_model.resolve()) if args.second_model else None,
    }
    predictor = make_ultralytics_predictor(args.model, configuration, args.second_model)
    manifest = build_package(
        args.image_dir,
        args.label_dir,
        args.model,
        args.output_dir,
        predictor,
        configuration,
        second_model_path=args.second_model,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
