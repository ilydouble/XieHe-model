#!/usr/bin/env python3
"""Compare two 20-class, four-corner YOLO Pose models on one labelled test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CORNER_NAMES = ("TL", "TR", "BR", "BL")
BASE_CLASS_IDS = frozenset(range(18))
OPTIONAL_CLASS_IDS = frozenset({18, 19})
ALL_CLASS_IDS = BASE_CLASS_IDS | OPTIONAL_CLASS_IDS
PRIMARY_MODE = "native"
GT_COLOR = (45, 220, 90)
OLD_COLOR = (235, 70, 185)
NEW_COLOR = (35, 205, 235)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


@dataclass(frozen=True)
class CornerObject:
    class_id: int
    box_xyxy: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, float], ...]
    confidence: float = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def parse_corner_label(path: Path, width: int, height: int) -> dict[int, CornerObject]:
    objects: dict[int, CornerObject] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 17:
            raise ValueError(f"{path}:{line_number}: expected 17 fields, found {len(values)}")
        class_id = int(float(values[0]))
        if class_id in objects or class_id not in ALL_CLASS_IDS:
            raise ValueError(f"{path}:{line_number}: invalid or duplicate class {class_id}")
        cx, cy, box_w, box_h = (float(value) for value in values[1:5])
        box = (
            (cx - box_w / 2) * width,
            (cy - box_h / 2) * height,
            (cx + box_w / 2) * width,
            (cy + box_h / 2) * height,
        )
        points = []
        for index in range(5, 17, 3):
            x, y, visibility = (float(value) for value in values[index : index + 3])
            points.append((x * width, y * height, visibility))
        objects[class_id] = CornerObject(class_id, box, tuple(points))
    if not objects:
        raise ValueError(f"{path}: no objects")
    return objects


def find_pairs(image_dir: Path, label_dir: Path, limit: int | None = None) -> list[tuple[Path, Path]]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    missing = [path.name for path in images if path.stem not in labels]
    extra = sorted(set(labels) - {path.stem for path in images})
    if missing or extra:
        raise ValueError(f"image/label mismatch: missing={missing[:5]}, extra={extra[:5]}")
    pairs = [(path, labels[path.stem]) for path in images]
    return pairs if limit is None else pairs[:limit]


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / (first_area + second_area - intersection + 1e-9)


def native_assignments(predictions: Sequence[CornerObject], confidence: float) -> dict[int, CornerObject]:
    assigned: dict[int, CornerObject] = {}
    for prediction in predictions:
        if prediction.confidence < confidence:
            continue
        current = assigned.get(prediction.class_id)
        if current is None or prediction.confidence > current.confidence:
            assigned[prediction.class_id] = prediction
    return assigned


def production_assignments(
    predictions: Sequence[CornerObject], confidence: float = 0.5, iou_threshold: float = 0.3
) -> dict[int, CornerObject]:
    """Mirror 3-inference/app.py: confidence NMS, then assign V0.. by mean keypoint y."""
    candidates = sorted(
        (prediction for prediction in predictions if prediction.confidence >= confidence),
        key=lambda prediction: -prediction.confidence,
    )
    kept: list[CornerObject] = []
    for candidate in candidates:
        if not any(box_iou(candidate.box_xyxy, existing.box_xyxy) > iou_threshold for existing in kept):
            kept.append(candidate)
    kept.sort(key=lambda prediction: statistics.fmean(point[1] for point in prediction.keypoints))
    return {
        rank: CornerObject(rank, prediction.box_xyxy, prediction.keypoints, prediction.confidence)
        for rank, prediction in enumerate(kept)
    }


def evaluate_assignments(
    truth: dict[int, CornerObject], assigned: dict[int, CornerObject], width: int, height: int
) -> dict:
    diagonal = math.hypot(width, height)
    point_rows = []
    for class_id in sorted(truth):
        expected = truth[class_id]
        predicted = assigned.get(class_id)
        for corner_index, (gt_x, gt_y, visibility) in enumerate(expected.keypoints):
            row = {
                "class_id": class_id,
                "vertebra": f"V{class_id}",
                "corner": CORNER_NAMES[corner_index],
                "visible": visibility > 0,
                "detected": predicted is not None,
                "distance_px": None,
                "distance_diagonal": None,
            }
            if visibility > 0 and predicted is not None and corner_index < len(predicted.keypoints):
                pred_x, pred_y, _ = predicted.keypoints[corner_index]
                distance = math.hypot(pred_x - gt_x, pred_y - gt_y)
                row["distance_px"] = round(distance, 4)
                row["distance_diagonal"] = round(distance / diagonal, 8)
            point_rows.append(row)
    visible = [row for row in point_rows if row["visible"]]
    measured = [row for row in visible if row["distance_px"] is not None]
    errors = [row["distance_px"] for row in measured]
    diag_errors = [row["distance_diagonal"] for row in measured]
    expected_classes = set(truth)
    assigned_classes = set(assigned)
    return {
        "truth_classes": sorted(expected_classes),
        "truth_vertebrae": len(expected_classes),
        "predicted_vertebrae": len(assigned_classes),
        "matched_vertebrae": len(expected_classes & assigned_classes),
        "missing_vertebrae": sorted(expected_classes - assigned_classes),
        "extra_vertebrae": sorted(assigned_classes - expected_classes),
        "complete_ground_truth": expected_classes.issubset(assigned_classes),
        "exact_ground_truth": assigned_classes == expected_classes,
        "complete_18": expected_classes == BASE_CLASS_IDS and BASE_CLASS_IDS.issubset(assigned_classes),
        "exactly_18": assigned_classes == BASE_CLASS_IDS,
        "visible_points": len(visible),
        "detected_points": len(measured),
        "mean_error_px": None if not errors else round(statistics.fmean(errors), 4),
        "max_error_px": None if not errors else round(max(errors), 4),
        "mean_error_diagonal": None if not diag_errors else round(statistics.fmean(diag_errors), 8),
        "pck_10_hits": sum(value <= 10 for value in errors),
        "pck_20_hits": sum(value <= 20 for value in errors),
        "point_rows": point_rows,
    }


def aggregate_mode(samples: Sequence[dict], model_key: str, mode: str) -> dict:
    metrics = [sample[model_key][mode] for sample in samples]
    points = [row for metric in metrics for row in metric["point_rows"] if row["visible"]]
    measured = [row for row in points if row["distance_px"] is not None]
    errors = [row["distance_px"] for row in measured]
    diag_errors = [row["distance_diagonal"] for row in measured]
    per_vertebra = {}
    for class_id in sorted(ALL_CLASS_IDS):
        rows = [row for row in points if row["class_id"] == class_id]
        found = [row for row in rows if row["distance_px"] is not None]
        per_vertebra[f"V{class_id}"] = {
            "detected_points": len(found),
            "total_points": len(rows),
            "mean_error_px": None if not found else round(statistics.fmean(row["distance_px"] for row in found), 3),
        }
    per_corner = {}
    for corner in CORNER_NAMES:
        rows = [row for row in points if row["corner"] == corner]
        found = [row for row in rows if row["distance_px"] is not None]
        per_corner[corner] = {
            "detected_points": len(found),
            "total_points": len(rows),
            "mean_error_px": None if not found else round(statistics.fmean(row["distance_px"] for row in found), 3),
        }
    return {
        "sample_count": len(samples),
        "visible_points": len(points),
        "detected_points": len(measured),
        "point_recall": round(len(measured) / len(points), 6) if points else None,
        "mean_error_px": round(statistics.fmean(errors), 3) if errors else None,
        "median_error_px": round(percentile(errors, 0.5), 3) if errors else None,
        "p90_error_px": round(percentile(errors, 0.9), 3) if errors else None,
        "mean_error_diagonal_pct": round(statistics.fmean(diag_errors) * 100, 4) if diag_errors else None,
        "pck_10_all": round(sum(value <= 10 for value in errors) / len(points), 6) if points else None,
        "pck_20_all": round(sum(value <= 20 for value in errors) / len(points), 6) if points else None,
        "complete_18_images": sum(metric["complete_18"] for metric in metrics),
        "exactly_18_images": sum(metric["exactly_18"] for metric in metrics),
        "complete_ground_truth_images": sum(metric["complete_ground_truth"] for metric in metrics),
        "exact_ground_truth_images": sum(metric["exact_ground_truth"] for metric in metrics),
        "mean_predicted_vertebrae": round(statistics.fmean(metric["predicted_vertebrae"] for metric in metrics), 3),
        "per_vertebra": per_vertebra,
        "per_corner": per_corner,
    }


def compare_samples(samples: Sequence[dict]) -> dict:
    comparable = [sample for sample in samples if sample["old"][PRIMARY_MODE]["mean_error_px"] is not None and sample["new"][PRIMARY_MODE]["mean_error_px"] is not None]
    deltas = [sample["old"][PRIMARY_MODE]["mean_error_px"] - sample["new"][PRIMARY_MODE]["mean_error_px"] for sample in comparable]
    return {
        "comparable_images": len(comparable),
        "new_improved_images": sum(delta > 0.05 for delta in deltas),
        "new_worsened_images": sum(delta < -0.05 for delta in deltas),
        "near_tie_images": sum(abs(delta) <= 0.05 for delta in deltas),
        "mean_improvement_px": round(statistics.fmean(deltas), 3) if deltas else None,
        "median_improvement_px": round(percentile(deltas, 0.5), 3) if deltas else None,
    }


def add_hybrid_metrics(samples: Sequence[dict], summary: dict) -> None:
    """Retain the legacy 18-class y-order diagnostic only for ordinary 18-class GT."""
    for model_key in ("old", "new"):
        for sample in samples:
            production = sample[model_key]["production"]
            native = sample[model_key]["native"]
            use_legacy_y_order = native["truth_classes"] == sorted(BASE_CLASS_IDS) and production["predicted_vertebrae"] == 18
            sample[model_key]["hybrid"] = production if use_legacy_y_order else native
        summary[model_key]["hybrid"] = aggregate_mode(samples, model_key, "hybrid")


def source_group(filename: str) -> str:
    if filename.startswith("eap_"):
        return "eap"
    if filename.startswith("1.2."):
        return "server_uid"
    if filename[:1].isdigit():
        return "legacy_numeric"
    return "new_site_code"


def aggregate_sources(samples: Sequence[dict]) -> dict:
    result = {}
    for group in sorted({source_group(sample["filename"]) for sample in samples}):
        rows = [sample for sample in samples if source_group(sample["filename"]) == group]
        old_errors = [sample["old"][PRIMARY_MODE]["mean_error_px"] for sample in rows]
        new_errors = [sample["new"][PRIMARY_MODE]["mean_error_px"] for sample in rows]
        deltas = [old - new for old, new in zip(old_errors, new_errors)]
        result[group] = {
            "sample_count": len(rows),
            "old_mean_image_error_px": round(statistics.fmean(old_errors), 3),
            "new_mean_image_error_px": round(statistics.fmean(new_errors), 3),
            "mean_improvement_px": round(statistics.fmean(deltas), 3),
            "new_improved_images": sum(delta > 0.05 for delta in deltas),
            "new_worsened_images": sum(delta < -0.05 for delta in deltas),
        }
    return result


def choose_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, max_width: int = 600, max_height: int = 1000) -> tuple[Image.Image, float]:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    if scale >= 1:
        return image.copy(), 1.0
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS), scale


def draw_objects(image: Image.Image, objects: Iterable[CornerObject], color: tuple[int, int, int], scale: float) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = choose_font(max(11, round(min(output.size) * 0.013)))
    radius = max(2, round(min(output.size) * 0.0035))
    for obj in objects:
        points = [(point[0] * scale, point[1] * scale) for point in obj.keypoints]
        if len(points) == 4:
            draw.line(points + [points[0]], fill=color, width=max(1, radius))
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=BLACK)
        if points:
            draw.text((points[0][0] + radius + 1, points[0][1] - radius), f"V{obj.class_id}", font=font, fill=color, stroke_width=2, stroke_fill=BLACK)
    return output


def add_title(image: Image.Image, title: str, color: tuple[int, int, int]) -> Image.Image:
    bar = 44
    canvas = Image.new("RGB", (image.width, image.height + bar), (24, 24, 24))
    canvas.paste(image, (0, bar))
    ImageDraw.Draw(canvas).text((10, 9), title, font=choose_font(21), fill=color)
    return canvas


def render_preview(image_path: Path, truth: dict[int, CornerObject], old: dict[int, CornerObject], new: dict[int, CornerObject], old_error, new_error, output_path: Path) -> None:
    with Image.open(image_path) as source:
        source.load()
        source = ImageOps.exif_transpose(source).convert("RGB")
        base, scale = fit_image(source)
    gt_panel = add_title(draw_objects(base, truth.values(), GT_COLOR, scale), f"GT: {len(truth)} vertebrae / {len(truth) * 4} corners", GT_COLOR)
    old_panel = draw_objects(base, truth.values(), GT_COLOR, scale)
    old_panel = add_title(draw_objects(old_panel, old.values(), OLD_COLOR, scale), f"Old: {old_error} px", OLD_COLOR)
    new_panel = draw_objects(base, truth.values(), GT_COLOR, scale)
    new_panel = add_title(draw_objects(new_panel, new.values(), NEW_COLOR, scale), f"New: {new_error} px", NEW_COLOR)
    gap = 8
    canvas = Image.new("RGB", (gt_panel.width * 3 + gap * 2, gt_panel.height), (70, 70, 70))
    canvas.paste(gt_panel, (0, 0)); canvas.paste(old_panel, (gt_panel.width + gap, 0)); canvas.paste(new_panel, ((gt_panel.width + gap) * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=88, optimize=True)


def predict_dataset(model_path: Path, pairs: Sequence[tuple[Path, Path]], imgsz: int, device: str, raw_confidence: float) -> tuple[dict[str, list[CornerObject]], dict, dict[str, float]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if pairs:
        model.predict(str(pairs[0][0]), imgsz=imgsz, conf=raw_confidence, device=device, verbose=False)
    predictions: dict[str, list[CornerObject]] = {}
    timings = []
    timing_by_file = {}
    for index, (image_path, _) in enumerate(pairs, 1):
        started = time.perf_counter()
        result = model.predict(str(image_path), imgsz=imgsz, conf=raw_confidence, device=device, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        timings.append(elapsed_ms)
        timing_by_file[image_path.name] = round(elapsed_ms, 3)
        objects = []
        if result.boxes is not None and result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            keypoints = result.keypoints.data.cpu().tolist()
            for box, confidence, class_id, points in zip(boxes, confidences, classes, keypoints):
                objects.append(CornerObject(int(class_id), tuple(float(value) for value in box), tuple(tuple(float(value) for value in point) for point in points), float(confidence)))
        predictions[image_path.name] = objects
        print(f"[{model_path.parent.parent.name} {index}/{len(pairs)}] {image_path.name}: raw={len(objects)}", flush=True)
    return predictions, {
        "mean_ms": round(statistics.fmean(timings), 3),
        "median_ms": round(percentile(timings, 0.5), 3),
        "p90_ms": round(percentile(timings, 0.9), 3),
    }, timing_by_file


def official_test_metrics(model_path: Path, data_yaml: Path, imgsz: int, device: str, project: Path) -> dict:
    from ultralytics import YOLO
    import ultralytics.data.dataset as dataset_module

    original_saver = dataset_module.save_dataset_cache_file

    def no_disk_cache(prefix, path, data, version):
        data["version"] = version

    dataset_module.save_dataset_cache_file = no_disk_cache
    try:
        metrics = YOLO(str(model_path)).val(
            data=str(data_yaml.resolve()), split="test", imgsz=imgsz, batch=1, device=device,
            conf=0.001, iou=0.7, max_det=300, plots=False, save=False, verbose=False,
            project=str(project), name=model_path.parent.parent.name,
        )
    finally:
        dataset_module.save_dataset_cache_file = original_saver

    def metric_block(value) -> dict:
        return {
            "precision": round(float(value.mp), 6), "recall": round(float(value.mr), 6),
            "map50": round(float(value.map50), 6), "map75": round(float(value.map75), 6),
            "map50_95": round(float(value.map), 6),
        }

    return {"box": metric_block(metrics.box), "pose": metric_block(metrics.pose), "speed_ms": {key: round(float(value), 3) for key, value in metrics.speed.items()}}


def write_csv(path: Path, samples: Sequence[dict]) -> None:
    fields = ["filename", "old_error_px", "new_error_px", "new_improvement_px", "old_predicted", "new_predicted", "old_native_error_px", "new_native_error_px", "old_ms", "new_ms", "preview"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for sample in samples:
            old_prod, new_prod = sample["old"][PRIMARY_MODE], sample["new"][PRIMARY_MODE]
            writer.writerow({
                "filename": sample["filename"], "old_error_px": old_prod["mean_error_px"], "new_error_px": new_prod["mean_error_px"],
                "new_improvement_px": sample["new_improvement_px"], "old_predicted": old_prod["predicted_vertebrae"], "new_predicted": new_prod["predicted_vertebrae"],
                "old_native_error_px": sample["old"]["native"]["mean_error_px"], "new_native_error_px": sample["new"]["native"]["mean_error_px"],
                "old_ms": sample["old"]["inference_ms"], "new_ms": sample["new"]["inference_ms"], "preview": sample["preview"],
            })


HTML = """<!doctype html><meta charset=utf-8><title>Corner新旧模型对比</title><style>*{box-sizing:border-box}body{margin:0;background:#11151b;color:#edf2f7;font-family:-apple-system,sans-serif}.top{position:sticky;top:0;background:#18202bee;padding:12px 18px}.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button,select,input{background:#202b38;color:#fff;border:1px solid #46566a;border-radius:7px;padding:7px 9px}.main{max-width:1900px;margin:auto;padding:15px}.card{background:#1a222d;border:1px solid #34404f;border-radius:10px;padding:14px}.preview{width:100%}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:10px 0}.metric{background:#111821;padding:9px;border-radius:7px}.good{color:#65e097}.bad{color:#ff8080}</style><div class=top><div class=row><b>Corner新旧模型对比</b><select id=filter><option value=all>全部</option><option value=worse>新模型恶化</option><option value=better>新模型改善</option><option value=incomplete>18节不完整</option></select><select id=sort><option value=worse>按恶化最多</option><option value=better>按改善最多</option><option value=newbad>按新模型误差</option></select><button id=prev>←</button><button id=next>→</button><input id=jump type=number min=1 style=width:80px></div><div id=stats></div></div><main class=main><section class=card><h2 id=title></h2><img class=preview id=preview><div class=meta id=meta></div><div id=position></div></section></main><script src=review_data.js></script><script>(()=>{const d=window.CORNER_COMPARISON,s=d.samples,$=x=>document.getElementById(x);let ids=[],at=0;function refresh(){ids=s.map((x,i)=>i).filter(i=>$('filter').value==='all'||($('filter').value==='worse'&&s[i].new_improvement_px<0)||($('filter').value==='better'&&s[i].new_improvement_px>0)||($('filter').value==='incomplete'&&(!s[i].old.production.exactly_18||!s[i].new.production.exactly_18)));ids.sort((a,b)=>$('sort').value==='better'?s[b].new_improvement_px-s[a].new_improvement_px:$('sort').value==='newbad'?s[b].new.production.mean_error_px-s[a].new.production.mean_error_px:s[a].new_improvement_px-s[b].new_improvement_px);at=Math.min(at,Math.max(0,ids.length-1));render()}function render(){const x=s[ids[at]];if(!x)return;$('stats').textContent=`${d.summary.sample_count}张：新模型改善 ${d.comparison.new_improved_images}，恶化 ${d.comparison.new_worsened_images}，平均改善 ${d.comparison.mean_improvement_px}px`;$('title').textContent=x.filename;$('preview').src=x.preview;const delta=x.new_improvement_px,cls=delta>=0?'good':'bad';$('meta').innerHTML=`<div class=metric>旧误差 <b>${x.old.production.mean_error_px}px</b></div><div class=metric>新误差 <b>${x.new.production.mean_error_px}px</b></div><div class=metric>改善 <b class=${cls}>${delta}px</b></div><div class=metric>旧/新检出 <b>${x.old.production.predicted_vertebrae}/${x.new.production.predicted_vertebrae}</b></div><div class=metric>旧/新耗时 <b>${x.old.inference_ms}/${x.new.inference_ms}ms</b></div>`;$('position').textContent=`${at+1}/${ids.length}`;$('jump').value=at+1}function move(n){at=Math.max(0,Math.min(ids.length-1,at+n));render()}$('filter').onchange=()=>{at=0;refresh()};$('sort').onchange=()=>{at=0;refresh()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('jump').onchange=()=>{at=Math.max(0,Math.min(ids.length-1,Number($('jump').value)-1));render()};document.onkeydown=e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)};refresh()})()</script>"""

# The 20-class contract must preserve semantic model classes.  In particular,
# class 19 is T13 and is not the bottom-most vertebra, so y-rank is diagnostic only.
HTML = """<!doctype html><meta charset=utf-8><title>Corner 20类模型对比</title><style>*{box-sizing:border-box}body{margin:0;background:#11151b;color:#edf2f7;font-family:-apple-system,sans-serif}.top{position:sticky;top:0;background:#18202bee;padding:12px 18px}.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button,select,input{background:#202b38;color:#fff;border:1px solid #46566a;border-radius:7px;padding:7px 9px}.main{max-width:1900px;margin:auto;padding:15px}.card{background:#1a222d;border:1px solid #34404f;border-radius:10px;padding:14px}.preview{width:100%}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:10px 0}.metric{background:#111821;padding:9px;border-radius:7px}.good{color:#65e097}.bad{color:#ff8080}</style><div class=top><div class=row><b>Corner 20类语义对比</b><select id=filter><option value=all>全部</option><option value=worse>新模型恶化</option><option value=better>新模型改善</option><option value=incomplete>未完整匹配GT类别</option></select><select id=sort><option value=worse>按恶化最多</option><option value=better>按改善最多</option><option value=newbad>按新模型误差</option></select><button id=prev>←</button><button id=next>→</button><input id=jump type=number min=1 style=width:80px></div><div id=stats></div></div><main class=main><section class=card><h2 id=title></h2><img class=preview id=preview><div class=meta id=meta></div><div id=position></div></section></main><script src=review_data.js></script><script>(()=>{const d=window.CORNER_COMPARISON,s=d.samples,$=x=>document.getElementById(x),m='native';let ids=[],at=0;function refresh(){ids=s.map((x,i)=>i).filter(i=>$('filter').value==='all'||($('filter').value==='worse'&&s[i].new_improvement_px<0)||($('filter').value==='better'&&s[i].new_improvement_px>0)||($('filter').value==='incomplete'&&(!s[i].old[m].complete_ground_truth||!s[i].new[m].complete_ground_truth)));ids.sort((a,b)=>$('sort').value==='better'?s[b].new_improvement_px-s[a].new_improvement_px:$('sort').value==='newbad'?s[b].new[m].mean_error_px-s[a].new[m].mean_error_px:s[a].new_improvement_px-s[b].new_improvement_px);at=Math.min(at,Math.max(0,ids.length-1));render()}function render(){const x=s[ids[at]];if(!x)return;$('stats').textContent=`${d.summary.sample_count}张：新模型改善 ${d.comparison.new_improved_images}，恶化 ${d.comparison.new_worsened_images}，平均改善 ${d.comparison.mean_improvement_px}px`;$('title').textContent=x.filename;$('preview').src=x.preview;const delta=x.new_improvement_px,cls=delta>=0?'good':'bad';$('meta').innerHTML=`<div class=metric>旧误差 <b>${x.old[m].mean_error_px}px</b></div><div class=metric>新误差 <b>${x.new[m].mean_error_px}px</b></div><div class=metric>改善 <b class=${cls}>${delta}px</b></div><div class=metric>旧/新检出 <b>${x.old[m].predicted_vertebrae}/${x.new[m].predicted_vertebrae}</b></div><div class=metric>旧/新耗时 <b>${x.old.inference_ms}/${x.new.inference_ms}ms</b></div>`;$('position').textContent=`${at+1}/${ids.length}`;$('jump').value=at+1}function move(n){at=Math.max(0,Math.min(ids.length-1,at+n));render()}$('filter').onchange=()=>{at=0;refresh()};$('sort').onchange=()=>{at=0;refresh()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('jump').onchange=()=>{at=Math.max(0,Math.min(ids.length-1,at+Number($('jump').value)-1-at));render()};document.onkeydown=e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)};refresh()})()</script>"""


def write_report(path: Path, manifest: dict) -> None:
    old, new = manifest["summary"]["old"], manifest["summary"]["new"]
    comparison = manifest["comparison"]
    source_rows = "\n".join(
        f"| {group} | {values['sample_count']} | {values['old_mean_image_error_px']} | {values['new_mean_image_error_px']} | {values['mean_improvement_px']} | {values['new_improved_images']}/{values['new_worsened_images']} |"
        for group, values in manifest["source_comparison"].items()
    )
    vertebra_rows = "\n".join(
        f"| {name} | {old[PRIMARY_MODE]['per_vertebra'][name]['mean_error_px']} | {new[PRIMARY_MODE]['per_vertebra'][name]['mean_error_px']} | {None if old[PRIMARY_MODE]['per_vertebra'][name]['mean_error_px'] is None or new[PRIMARY_MODE]['per_vertebra'][name]['mean_error_px'] is None else round(old[PRIMARY_MODE]['per_vertebra'][name]['mean_error_px'] - new[PRIMARY_MODE]['per_vertebra'][name]['mean_error_px'], 3)} |"
        for name in old[PRIMARY_MODE]["per_vertebra"]
    )
    official = manifest["official_test"]
    official_section = ""
    if official is not None:
        official_old, official_new = official["old"], official["new"]
        official_section = f"""
## 同一test官方Ultralytics指标

| 指标 | 旧模型 | 新模型 |
|---|---:|---:|
| Box mAP50-95 | {official_old['box']['map50_95']:.4f} | {official_new['box']['map50_95']:.4f} |
| Pose mAP50 | {official_old['pose']['map50']:.4f} | {official_new['pose']['map50']:.4f} |
| Pose mAP50-95 | {official_old['pose']['map50_95']:.4f} | {official_new['pose']['map50_95']:.4f} |
| Pose precision | {official_old['pose']['precision']:.4f} | {official_new['pose']['precision']:.4f} |
| Pose recall | {official_old['pose']['recall']:.4f} | {official_new['pose']['recall']:.4f} |
"""
    content = f"""# Corner新旧模型同test对比

- test：{manifest['summary']['sample_count']}张；普通病例为V0–V17，变异病例可选含V18/L6或V19/T13
- 旧模型：`{manifest['models']['old']['path']}`
- 新模型：`{manifest['models']['new']['path']}`
- 推理：imgsz={manifest['configuration']['imgsz']}，conf={manifest['configuration']['confidence']}；主指标保留模型原始类别ID

## 20类语义主结果（native class）

| 指标 | 旧模型 | 新模型 |
|---|---:|---:|
| 标注点召回 | {old[PRIMARY_MODE]['point_recall']:.2%} | {new[PRIMARY_MODE]['point_recall']:.2%} |
| 平均误差(px) | {old[PRIMARY_MODE]['mean_error_px']} | {new[PRIMARY_MODE]['mean_error_px']} |
| 中位误差(px) | {old[PRIMARY_MODE]['median_error_px']} | {new[PRIMARY_MODE]['median_error_px']} |
| P90误差(px) | {old[PRIMARY_MODE]['p90_error_px']} | {new[PRIMARY_MODE]['p90_error_px']} |
| PCK@10(含漏点失败) | {old[PRIMARY_MODE]['pck_10_all']:.2%} | {new[PRIMARY_MODE]['pck_10_all']:.2%} |
| PCK@20(含漏点失败) | {old[PRIMARY_MODE]['pck_20_all']:.2%} | {new[PRIMARY_MODE]['pck_20_all']:.2%} |
| 完整匹配GT类别图像 | {old[PRIMARY_MODE]['exact_ground_truth_images']}/{manifest['summary']['sample_count']} | {new[PRIMARY_MODE]['exact_ground_truth_images']}/{manifest['summary']['sample_count']} |
| 自定义单图推理均值(ms) | {old['timing']['mean_ms']} | {new['timing']['mean_ms']} |

## 旧18类y排序兼容诊断

仅当GT严格为V0–V17且y排序恰好得到18节时沿用旧逻辑；含L6/T13时必须使用模型原始类别。V19/T13位于T12与L1之间，不能按全局y排序得到类别19。

| 指标 | 旧模型 | 新模型 |
|---|---:|---:|
| 平均误差(px) | {old['hybrid']['mean_error_px']} | {new['hybrid']['mean_error_px']} |
| P90误差(px) | {old['hybrid']['p90_error_px']} | {new['hybrid']['p90_error_px']} |
| PCK@20(含漏点失败) | {old['hybrid']['pck_20_all']:.2%} | {new['hybrid']['pck_20_all']:.2%} |
| 标注点召回 | {old['hybrid']['point_recall']:.2%} | {new['hybrid']['point_recall']:.2%} |

新模型逐图改善{comparison['new_improved_images']}张、恶化{comparison['new_worsened_images']}张、近似持平{comparison['near_tie_images']}张；平均改善为{comparison['mean_improvement_px']} px（正数表示新模型更好）。

## 来源分组（逐图平均）

| 来源 | 数量 | 旧误差(px) | 新误差(px) | 改善(px) | 改善/恶化张数 |
|---|---:|---:|---:|---:|---:|
{source_rows}

## 椎体类别分组

| 椎体 | 旧误差(px) | 新误差(px) | 改善(px) |
|---|---:|---:|---:|
{vertebra_rows}

{official_section}

`打开对比页面.html`可按改善、恶化、GT类别完整率和误差逐图查看。绿色为GT，洋红为旧模型，青色为新模型。
"""
    path.write_text(content, encoding="utf-8")


def package_file_hashes(output_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def finalize_package(output_dir: Path, manifest: dict) -> None:
    add_hybrid_metrics(manifest["samples"], manifest["summary"])
    manifest["source_comparison"] = aggregate_sources(manifest["samples"])
    data = {"summary": manifest["summary"], "comparison": manifest["comparison"], "samples": manifest["samples"]}
    (output_dir / "review_data.js").write_text("window.CORNER_COMPARISON=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    write_report(output_dir / "对比报告.md", manifest)
    manifest["package_files"] = package_file_hashes(output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_comparison(args: argparse.Namespace) -> dict:
    pairs = find_pairs(args.image_dir, args.label_dir, args.limit)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    (args.output_dir / "previews").mkdir(parents=True)
    old_predictions, old_timing, old_timing_by_file = predict_dataset(args.old_model, pairs, args.imgsz, args.device, args.raw_conf)
    new_predictions, new_timing, new_timing_by_file = predict_dataset(args.new_model, pairs, args.imgsz, args.device, args.raw_conf)
    samples = []
    for index, (image_path, label_path) in enumerate(pairs, 1):
        with Image.open(image_path) as image:
            image.load(); width, height = ImageOps.exif_transpose(image).size
        truth = parse_corner_label(label_path, width, height)
        old_raw, new_raw = old_predictions[image_path.name], new_predictions[image_path.name]
        old_native, new_native = native_assignments(old_raw, args.confidence), native_assignments(new_raw, args.confidence)
        old_production = production_assignments(old_raw, args.confidence, args.nms_iou)
        new_production = production_assignments(new_raw, args.confidence, args.nms_iou)
        old_metrics = {"native": evaluate_assignments(truth, old_native, width, height), "production": evaluate_assignments(truth, old_production, width, height)}
        new_metrics = {"native": evaluate_assignments(truth, new_native, width, height), "production": evaluate_assignments(truth, new_production, width, height)}
        improvement = round(old_metrics[PRIMARY_MODE]["mean_error_px"] - new_metrics[PRIMARY_MODE]["mean_error_px"], 4)
        preview = f"previews/{index:04d}_{image_path.stem}.jpg"
        render_preview(image_path, truth, old_native, new_native, old_metrics[PRIMARY_MODE]["mean_error_px"], new_metrics[PRIMARY_MODE]["mean_error_px"], args.output_dir / preview)
        samples.append({
            "filename": image_path.name, "width": width, "height": height, "image_sha256": sha256_file(image_path), "label_sha256": sha256_file(label_path),
            "preview": preview, "new_improvement_px": improvement,
            "old": {**old_metrics, "inference_ms": old_timing_by_file[image_path.name]},
            "new": {**new_metrics, "inference_ms": new_timing_by_file[image_path.name]},
        })
    summary = {
        "sample_count": len(samples),
        "old": {"native": aggregate_mode(samples, "old", "native"), "production": aggregate_mode(samples, "old", "production"), "timing": old_timing},
        "new": {"native": aggregate_mode(samples, "new", "native"), "production": aggregate_mode(samples, "new", "production"), "timing": new_timing},
    }
    comparison = compare_samples(samples)
    official = None if args.skip_official else {
        "old": official_test_metrics(args.old_model, args.data_yaml, args.imgsz, args.device, args.output_dir / "official_val"),
        "new": official_test_metrics(args.new_model, args.data_yaml, args.imgsz, args.device, args.output_dir / "official_val"),
    }
    manifest = {
        "configuration": {"imgsz": args.imgsz, "device": args.device, "raw_conf": args.raw_conf, "confidence": args.confidence, "nms_iou": args.nms_iou, "primary_mode": PRIMARY_MODE},
        "models": {
            "old": {"path": str(args.old_model.resolve()), "sha256": sha256_file(args.old_model)},
            "new": {"path": str(args.new_model.resolve()), "sha256": sha256_file(args.new_model)},
        },
        "image_dir": str(args.image_dir.resolve()), "label_dir": str(args.label_dir.resolve()),
        "summary": summary, "comparison": comparison, "official_test": official, "samples": samples,
    }
    (args.output_dir / "打开对比页面.html").write_text(HTML, encoding="utf-8")
    write_csv(args.output_dir / "逐图对比.csv", samples)
    finalize_package(args.output_dir, manifest)
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/pose_corner_data/images/test"))
    parser.add_argument("--label-dir", type=Path, default=Path("datasets/pose_corner_data/labels/test"))
    parser.add_argument("--data-yaml", type=Path, default=Path("6-train_ap_model/corner_data.yaml"))
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-conf", type=float, default=0.001)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-official", action="store_true", help="skip Ultralytics mAP pass; useful only for smoke tests")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = build_comparison(args)
    print(json.dumps({"summary": manifest["summary"], "comparison": manifest["comparison"], "official_test": manifest["official_test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
