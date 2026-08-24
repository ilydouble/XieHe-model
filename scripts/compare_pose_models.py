#!/usr/bin/env python3
"""Compare two six-keypoint Pose weights with the XieHe online inference contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
from PIL import Image, ImageDraw, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_six_point_model_review import (  # noqa: E402
    GT_COLOR,
    KEYPOINT_NAMES,
    PoseLabel,
    add_title,
    draw_points,
    find_pairs,
    fit_image,
    parse_pose_label,
    sha256_file,
)
from build_two_stage_pose_review import (  # noqa: E402
    aggregate_stage,
    calculate_stage_metrics,
    distribution,
    draw_error_lines,
    rounded,
    source_group,
)
from two_stage_pose_inference import PosePrediction, empty_prediction, run_pose_model  # noqa: E402


OLD_COLOR = (235, 65, 185)
NEW_COLOR = (45, 195, 240)
WHITE = (255, 255, 255)
LR_SWAP = (1, 0, 3, 2, 5, 4)


def apply_lr_swap(prediction: PosePrediction) -> PosePrediction:
    """Map a legacy CR/CL, IR/IL, SR/SL output into normalized semantics."""
    return PosePrediction(
        image_width=prediction.image_width,
        image_height=prediction.image_height,
        box_xyxy=prediction.box_xyxy,
        box_confidence=prediction.box_confidence,
        keypoints_xy=tuple(prediction.keypoints_xy[index] for index in LR_SWAP),
        keypoint_confidences=tuple(prediction.keypoint_confidences[index] for index in LR_SWAP),
    )


def online_accept_prediction(
    prediction: PosePrediction,
    minimum_box_confidence: float = 0.5,
    minimum_x_span_fraction: float = 0.10,
    minimum_y_span_fraction: float = 0.20,
) -> tuple[PosePrediction, str | None]:
    """Reproduce XieHe AP's post-predict confidence and collapse rejection."""
    if prediction.box_xyxy is None or prediction.box_confidence < minimum_box_confidence:
        return empty_prediction(prediction.image_width, prediction.image_height), "box_confidence"
    if len(prediction.keypoints_xy) != 6:
        return empty_prediction(prediction.image_width, prediction.image_height), "keypoint_count"
    xs = [point[0] for point in prediction.keypoints_xy]
    ys = [point[1] for point in prediction.keypoints_xy]
    if max(xs) - min(xs) < prediction.image_width * minimum_x_span_fraction:
        return empty_prediction(prediction.image_width, prediction.image_height), "collapsed_x"
    if max(ys) - min(ys) < prediction.image_height * minimum_y_span_fraction:
        return empty_prediction(prediction.image_width, prediction.image_height), "collapsed_y"
    return prediction, None


def truth_span_px(label: PoseLabel, height: int) -> float:
    shoulders = statistics.mean(label.keypoints[index][1] * height for index in (0, 1))
    lower = statistics.mean(label.keypoints[index][1] * height for index in (2, 3, 4, 5))
    return lower - shoulders


def predict_dataset(
    model_path: Path,
    pairs: Sequence[tuple[Path, Path]],
    image_size: int,
    raw_confidence: float,
    minimum_box_confidence: float,
    device: str,
    warmup: int,
    legacy_lr_swap: bool,
) -> tuple[dict[str, PosePrediction], dict[str, str | None], dict[str, float], dict]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    def infer(image_path: Path) -> tuple[PosePrediction, str | None, float]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode image: {image_path}")
        started = time.perf_counter()
        prediction = run_pose_model(model, image, raw_confidence, image_size, device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if legacy_lr_swap:
            prediction = apply_lr_swap(prediction)
        accepted, rejection = online_accept_prediction(prediction, minimum_box_confidence)
        return accepted, rejection, elapsed_ms

    for _ in range(warmup):
        infer(pairs[0][0])
    predictions: dict[str, PosePrediction] = {}
    rejections: dict[str, str | None] = {}
    timings: dict[str, float] = {}
    for index, (image_path, _) in enumerate(pairs, 1):
        prediction, rejection, elapsed_ms = infer(image_path)
        predictions[image_path.name] = prediction
        rejections[image_path.name] = rejection
        timings[image_path.name] = rounded(elapsed_ms)
        print(
            f"[{model_path.parent.parent.name} {index}/{len(pairs)}] {image_path.name} "
            f"box={prediction.box_confidence:.4f} reject={rejection or '-'} {elapsed_ms:.1f}ms",
            flush=True,
        )
    return predictions, rejections, timings, distribution(list(timings.values()))


def normalized_points(prediction: PosePrediction) -> tuple[tuple[float, float, float], ...]:
    return prediction.normalized_keypoints()


def render_panel(
    base: Image.Image,
    label: PoseLabel,
    prediction: PosePrediction,
    metrics: dict,
    color: tuple[int, int, int],
    prefix: str,
    title: str,
) -> Image.Image:
    points = normalized_points(prediction)
    panel = draw_error_lines(base, label.keypoints, points, color)
    panel = draw_points(panel, label.keypoints, GT_COLOR, "GT-")
    panel = draw_points(panel, points, color, prefix)
    return add_title(panel, f"{title}  err={metrics['mean_error_px']}px", WHITE)


def render_preview(
    image_path: Path,
    label: PoseLabel,
    old_prediction: PosePrediction,
    new_prediction: PosePrediction,
    old_metrics: dict,
    new_metrics: dict,
    output_path: Path,
) -> tuple[int, int]:
    with Image.open(image_path) as source:
        source.load()
        source = ImageOps.exif_transpose(source).convert("RGB")
        original_size = source.size
        base, _ = fit_image(source, 800, 1100)
    truth = add_title(draw_points(base, label.keypoints, GT_COLOR, "GT-"), "Ground truth", GT_COLOR)
    old = render_panel(base, label, old_prediction, old_metrics, OLD_COLOR, "OLD-", "Online baseline")
    new = render_panel(base, label, new_prediction, new_metrics, NEW_COLOR, "NEW-", "Latest model")
    gap = 8
    canvas = Image.new("RGB", (truth.width * 3 + gap * 2, truth.height), (70, 70, 70))
    canvas.paste(truth, (0, 0))
    canvas.paste(old, (truth.width + gap, 0))
    canvas.paste(new, ((truth.width + gap) * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=90, optimize=True)
    return original_size


def aggregate_model(samples: Sequence[dict], model_key: str, timing: dict) -> dict:
    result = aggregate_stage(samples, f"{model_key}_metrics")
    result["pck_10px_all"] = rounded(
        sum(
            point["distance_px"] is not None and point["distance_px"] <= 10
            for sample in samples
            for point in sample[f"{model_key}_metrics"]["points"]
            if point["visible"]
        )
        / result["visible_points"],
        6,
    )
    result["pck_20px_all"] = rounded(
        sum(
            point["distance_px"] is not None and point["distance_px"] <= 20
            for sample in samples
            for point in sample[f"{model_key}_metrics"]["points"]
            if point["visible"]
        )
        / result["visible_points"],
        6,
    )
    shoulder_values = [sample[f"{model_key}_metrics"]["shoulder_mean_dy_px"] for sample in samples]
    lower_values = [sample[f"{model_key}_metrics"]["lower_mean_dy_px"] for sample in samples]
    span_ratios = [
        sample[f"{model_key}_metrics"]["span_bias_px"] / sample["truth_span_px"]
        for sample in samples
        if sample[f"{model_key}_metrics"]["span_bias_px"] is not None and sample["truth_span_px"] > 0
    ]
    valid_shoulders = [value for value in shoulder_values if value is not None]
    valid_lower = [value for value in lower_values if value is not None]
    result.update(
        {
            "shoulders_above_truth_images": sum(value < 0 for value in valid_shoulders),
            "shoulders_above_truth_fraction": rounded(
                None if not valid_shoulders else sum(value < 0 for value in valid_shoulders) / len(valid_shoulders), 6
            ),
            "lower_below_truth_images": sum(value > 0 for value in valid_lower),
            "lower_below_truth_fraction": rounded(
                None if not valid_lower else sum(value > 0 for value in valid_lower) / len(valid_lower), 6
            ),
            "mean_span_ratio_bias": rounded(None if not span_ratios else statistics.mean(span_ratios), 6),
            "rejection_counts": {
                reason: sum(sample[f"{model_key}_rejection"] == reason for sample in samples)
                for reason in sorted({sample[f"{model_key}_rejection"] for sample in samples if sample[f"{model_key}_rejection"]})
            },
            "timing": timing,
        }
    )
    return result


def compare_samples(samples: Sequence[dict]) -> dict:
    comparable = [sample for sample in samples if sample["improvement_px"] is not None]
    improvements = [sample["improvement_px"] for sample in comparable]
    return {
        "comparable_images": len(comparable),
        "improved_images": sum(value > 0.05 for value in improvements),
        "worsened_images": sum(value < -0.05 for value in improvements),
        "near_tie_images": sum(abs(value) <= 0.05 for value in improvements),
        "mean_image_improvement_px": rounded(None if not improvements else statistics.mean(improvements)),
        "median_image_improvement_px": rounded(None if not improvements else statistics.median(improvements)),
    }


def aggregate_sources(samples: Sequence[dict]) -> dict:
    output = {}
    for source in sorted({sample["source"] for sample in samples}):
        group = [sample for sample in samples if sample["source"] == source]
        old_values = [sample["old_metrics"]["mean_error_px"] for sample in group if sample["old_metrics"]["mean_error_px"] is not None]
        new_values = [sample["new_metrics"]["mean_error_px"] for sample in group if sample["new_metrics"]["mean_error_px"] is not None]
        changes = [sample["improvement_px"] for sample in group if sample["improvement_px"] is not None]
        output[source] = {
            "sample_count": len(group),
            "old_mean_image_error_px": rounded(None if not old_values else statistics.mean(old_values)),
            "new_mean_image_error_px": rounded(None if not new_values else statistics.mean(new_values)),
            "mean_image_improvement_px": rounded(None if not changes else statistics.mean(changes)),
            "improved_images": sum(value > 0.05 for value in changes),
            "worsened_images": sum(value < -0.05 for value in changes),
        }
    return output


def write_csv(path: Path, samples: Sequence[dict]) -> None:
    fields = (
        "filename", "source", "old_error_px", "new_error_px", "improvement_px",
        "old_shoulder_dy_px", "new_shoulder_dy_px", "old_lower_dy_px", "new_lower_dy_px",
        "old_span_bias_px", "new_span_bias_px", "old_rejection", "new_rejection",
        "old_inference_ms", "new_inference_ms", "preview",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            old, new = sample["old_metrics"], sample["new_metrics"]
            writer.writerow(
                {
                    "filename": sample["filename"], "source": sample["source"],
                    "old_error_px": old["mean_error_px"], "new_error_px": new["mean_error_px"],
                    "improvement_px": sample["improvement_px"],
                    "old_shoulder_dy_px": old["shoulder_mean_dy_px"], "new_shoulder_dy_px": new["shoulder_mean_dy_px"],
                    "old_lower_dy_px": old["lower_mean_dy_px"], "new_lower_dy_px": new["lower_mean_dy_px"],
                    "old_span_bias_px": old["span_bias_px"], "new_span_bias_px": new["span_bias_px"],
                    "old_rejection": sample["old_rejection"] or "", "new_rejection": sample["new_rejection"] or "",
                    "old_inference_ms": sample["old_inference_ms"], "new_inference_ms": sample["new_inference_ms"],
                    "preview": sample["preview"],
                }
            )


HTML = r'''<!doctype html><meta charset="utf-8"><title>六点Pose线上版与最新版对比</title><style>*{box-sizing:border-box}body{margin:0;background:#11151b;color:#edf2f7;font-family:-apple-system,sans-serif}.top{position:sticky;top:0;background:#18202bee;padding:12px 18px}.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button,select,input{background:#202b38;color:#fff;border:1px solid #46566a;border-radius:7px;padding:7px 9px}.main{max-width:1900px;margin:auto;padding:15px}.card{background:#1a222d;border:1px solid #34404f;border-radius:10px;padding:14px}.preview{width:100%}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:10px 0}.metric{background:#111821;padding:9px;border-radius:7px}.good{color:#65e097}.bad{color:#ff8080}</style><div class=top><div class=row><b>六点Pose线上基线 vs 最新版</b><select id=filter><option value=all>全部</option><option value=worse>最新版恶化</option><option value=better>最新版改善</option><option value=reject>存在拒绝</option></select><select id=sort><option value=worse>恶化最多</option><option value=better>改善最多</option><option value=newbad>新误差最大</option></select><button id=prev>←</button><button id=next>→</button><input id=jump type=number min=1 style=width:80px></div><div id=stats></div></div><main class=main><section class=card><h2 id=title></h2><img class=preview id=preview><div class=meta id=meta></div><div id=position></div></section></main><script src=review_data.js></script><script>(()=>{const d=window.POSE_COMPARISON,s=d.samples,$=x=>document.getElementById(x);let ids=[],at=0;function refresh(){ids=s.map((x,i)=>i).filter(i=>$('filter').value==='all'||($('filter').value==='worse'&&s[i].improvement_px<0)||($('filter').value==='better'&&s[i].improvement_px>0)||($('filter').value==='reject'&&(s[i].old_rejection||s[i].new_rejection)));ids.sort((a,b)=>$('sort').value==='better'?(s[b].improvement_px??-1e9)-(s[a].improvement_px??-1e9):$('sort').value==='newbad'?(s[b].new_metrics.mean_error_px??1e9)-(s[a].new_metrics.mean_error_px??1e9):(s[a].improvement_px??1e9)-(s[b].improvement_px??1e9));at=Math.min(at,Math.max(0,ids.length-1));render()}function render(){const x=s[ids[at]];if(!x)return;$('stats').textContent=`${d.summary.sample_count}张：旧 ${d.summary.old.mean_error_px}px → 新 ${d.summary.new.mean_error_px}px；改善/恶化 ${d.comparison.improved_images}/${d.comparison.worsened_images}`;$('title').textContent=x.filename;$('preview').src=x.preview;const delta=x.improvement_px,cls=delta>=0?'good':'bad';$('meta').innerHTML=`<div class=metric>旧误差 <b>${x.old_metrics.mean_error_px}px</b></div><div class=metric>新误差 <b>${x.new_metrics.mean_error_px}px</b></div><div class=metric>改善 <b class=${cls}>${delta}px</b></div><div class=metric>肩dy <b>${x.old_metrics.shoulder_mean_dy_px} → ${x.new_metrics.shoulder_mean_dy_px}px</b></div><div class=metric>下四点dy <b>${x.old_metrics.lower_mean_dy_px} → ${x.new_metrics.lower_mean_dy_px}px</b></div><div class=metric>耗时 <b>${x.old_inference_ms}/${x.new_inference_ms}ms</b></div>`;$('position').textContent=`${at+1}/${ids.length}`;$('jump').value=at+1}function move(n){at=Math.max(0,Math.min(ids.length-1,at+n));render()}$('filter').onchange=()=>{at=0;refresh()};$('sort').onchange=()=>{at=0;refresh()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('jump').onchange=()=>{at=Math.max(0,Math.min(ids.length-1,Number($('jump').value)-1));render()};document.onkeydown=e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)};refresh()})()</script>'''


def metric_table(old: dict, new: dict) -> str:
    rows = (
        ("点召回", f"{old['detected_points']}/{old['visible_points']}", f"{new['detected_points']}/{new['visible_points']}"),
        ("平均误差(px)", old["mean_error_px"], new["mean_error_px"]),
        ("中位误差(px)", old["median_error_px"], new["median_error_px"]),
        ("P90误差(px)", old["p90_error_px"], new["p90_error_px"]),
        ("PCK@10(含漏点失败)", f"{old['pck_10px_all']:.2%}", f"{new['pck_10px_all']:.2%}"),
        ("PCK@20(含漏点失败)", f"{old['pck_20px_all']:.2%}", f"{new['pck_20px_all']:.2%}"),
        ("肩点平均dy(px)", old["shoulder_mean_dy_px"], new["shoulder_mean_dy_px"]),
        ("肩点高于标注图像", f"{old['shoulders_above_truth_fraction']:.2%}", f"{new['shoulders_above_truth_fraction']:.2%}"),
        ("下四点平均dy(px)", old["lower_mean_dy_px"], new["lower_mean_dy_px"]),
        ("下四点低于标注图像", f"{old['lower_below_truth_fraction']:.2%}", f"{new['lower_below_truth_fraction']:.2%}"),
        ("纵向跨度偏差(px)", old["mean_span_bias_px"], new["mean_span_bias_px"]),
        ("纵向跨度相对偏差", f"{old['mean_span_ratio_bias']:.2%}", f"{new['mean_span_ratio_bias']:.2%}"),
        ("CPU单图推理均值(ms)", old["timing"]["mean_ms"], new["timing"]["mean_ms"]),
    )
    return "\n".join(["| 指标 | 线上基线 | 最新版 |", "|---|---:|---:|", *(f"| {name} | {a} | {b} |" for name, a, b in rows)])


def write_report(path: Path, manifest: dict) -> None:
    old, new = manifest["summary"]["old"], manifest["summary"]["new"]
    comparison = manifest["comparison"]
    source_rows = "\n".join(
        f"| {name} | {values['sample_count']} | {values['old_mean_image_error_px']} | {values['new_mean_image_error_px']} | {values['mean_image_improvement_px']} | {values['improved_images']}/{values['worsened_images']} |"
        for name, values in manifest["by_source"].items()
    )
    point_rows = "\n".join(
        f"| {name} | {old['per_keypoint'][name]['mean_error_px']} | {new['per_keypoint'][name]['mean_error_px']} | {old['per_keypoint'][name]['mean_dy_px']} | {new['per_keypoint'][name]['mean_dy_px']} |"
        for name in KEYPOINT_NAMES
    )
    conclusion = "最新版总体变好" if new["mean_error_px"] < old["mean_error_px"] else "最新版总体没有变好"
    path.write_text(
        f"""# 六点Pose线上基线与最新版同test对比

结论：**{conclusion}**。点加权平均误差由{old['mean_error_px']} px变为{new['mean_error_px']} px，变化{rounded(old['mean_error_px'] - new['mean_error_px'])} px。

- test：患者隔离后的{manifest['summary']['sample_count']}张原始图，不裁黑边、不启用ROI二阶段。
- 线上基线：`{manifest['models']['old']['path']}`，SHA-256 `{manifest['models']['old']['sha256']}`。
- 最新版：`{manifest['models']['new']['path']}`，SHA-256 `{manifest['models']['new']['sha256']}`。
- 推理口径：XieHe原图单阶段，imgsz={manifest['configuration']['imgsz']}，候选conf={manifest['configuration']['raw_confidence']}，最终box conf≥{manifest['configuration']['minimum_box_confidence']}，并执行横/纵跨度塌缩拒绝。
- 权重边界：`xiehe-system`当前工作副本不含部署`pose.pt`，因此本报告的“线上基线”是同规范`best_performance-3`候选，不能替代对服务器权重SHA-256的最终核验。

## 总体指标

{metric_table(old, new)}

逐图改善{comparison['improved_images']}张、恶化{comparison['worsened_images']}张、近似持平{comparison['near_tie_images']}张；逐图平均改善{comparison['mean_image_improvement_px']} px，中位改善{comparison['median_image_improvement_px']} px。正数dy表示预测低于标注，负数表示预测高于标注。

## 来源分组（逐图平均误差）

| 来源 | 数量 | 基线误差px | 最新误差px | 改善px | 改善/恶化张数 |
|---|---:|---:|---:|---:|---:|
{source_rows}

## 六点分解

| 点 | 基线误差px | 最新误差px | 基线dy px | 最新dy px |
|---|---:|---:|---:|---:|
{point_rows}

`打开对比页面.html`可按最新版改善、恶化和误差逐张查看；绿色是真值，洋红是线上基线，青色是最新版。
""",
        encoding="utf-8",
    )


def package_hashes(output_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def build_comparison(args: argparse.Namespace) -> dict:
    pairs = find_pairs(args.image_dir, args.label_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    staging = args.output_dir.with_name(args.output_dir.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "previews").mkdir(parents=True)
    old_predictions, old_rejections, old_times, old_timing = predict_dataset(
        args.old_model, pairs, args.imgsz, args.raw_conf, args.minimum_box_confidence,
        args.device, args.warmup, args.old_legacy_lr_swap,
    )
    new_predictions, new_rejections, new_times, new_timing = predict_dataset(
        args.new_model, pairs, args.imgsz, args.raw_conf, args.minimum_box_confidence,
        args.device, args.warmup, args.new_legacy_lr_swap,
    )
    samples = []
    for index, (image_path, label_path) in enumerate(pairs, 1):
        label = parse_pose_label(label_path)
        old_prediction, new_prediction = old_predictions[image_path.name], new_predictions[image_path.name]
        width, height = old_prediction.image_width, old_prediction.image_height
        old_metrics = calculate_stage_metrics(label, old_prediction, width, height)
        new_metrics = calculate_stage_metrics(label, new_prediction, width, height)
        improvement = None
        if old_metrics["mean_error_px"] is not None and new_metrics["mean_error_px"] is not None:
            improvement = rounded(old_metrics["mean_error_px"] - new_metrics["mean_error_px"])
        preview = f"previews/{index:04d}_{image_path.stem}.jpg"
        render_preview(image_path, label, old_prediction, new_prediction, old_metrics, new_metrics, staging / preview)
        samples.append(
            {
                "filename": image_path.name, "source": source_group(image_path.name), "width": width, "height": height,
                "image_sha256": sha256_file(image_path), "label_sha256": sha256_file(label_path), "preview": preview,
                "truth_span_px": rounded(truth_span_px(label, height)), "improvement_px": improvement,
                "old_rejection": old_rejections[image_path.name], "new_rejection": new_rejections[image_path.name],
                "old_inference_ms": old_times[image_path.name], "new_inference_ms": new_times[image_path.name],
                "old_metrics": old_metrics, "new_metrics": new_metrics,
            }
        )
        print(f"[metrics {index}/{len(pairs)}] {image_path.name} old={old_metrics['mean_error_px']} new={new_metrics['mean_error_px']}", flush=True)
    summary = {
        "sample_count": len(samples),
        "old": aggregate_model(samples, "old", old_timing),
        "new": aggregate_model(samples, "new", new_timing),
    }
    manifest = {
        "configuration": {
            "imgsz": args.imgsz, "device": args.device, "raw_confidence": args.raw_conf,
            "minimum_box_confidence": args.minimum_box_confidence, "warmup": args.warmup,
            "old_legacy_lr_swap": args.old_legacy_lr_swap, "new_legacy_lr_swap": args.new_legacy_lr_swap,
            "pipeline": "XieHe full-image single-stage Pose with confidence and collapse rejection",
        },
        "models": {
            "old": {"path": str(args.old_model.resolve()), "sha256": sha256_file(args.old_model)},
            "new": {"path": str(args.new_model.resolve()), "sha256": sha256_file(args.new_model)},
        },
        "image_dir": str(args.image_dir.resolve()), "label_dir": str(args.label_dir.resolve()),
        "summary": summary, "comparison": compare_samples(samples), "by_source": aggregate_sources(samples), "samples": samples,
    }
    (staging / "打开对比页面.html").write_text(HTML, encoding="utf-8")
    write_csv(staging / "逐图对比.csv", samples)
    (staging / "review_data.js").write_text(
        "window.POSE_COMPARISON=" + json.dumps(
            {"summary": summary, "comparison": manifest["comparison"], "samples": samples},
            ensure_ascii=False, separators=(",", ":"),
        ) + ";\n",
        encoding="utf-8",
    )
    write_report(staging / "对比报告.md", manifest)
    manifest["package_files"] = package_hashes(staging)
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    staging.rename(args.output_dir)
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/pose_data/images/test"))
    parser.add_argument("--label-dir", type=Path, default=Path("datasets/pose_data/labels/test"))
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--raw-conf", type=float, default=0.25)
    parser.add_argument("--minimum-box-confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--old-legacy-lr-swap", action="store_true")
    parser.add_argument("--new-legacy-lr-swap", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = build_comparison(args)
    print(json.dumps({"summary": manifest["summary"], "comparison": manifest["comparison"], "by_source": manifest["by_source"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
