#!/usr/bin/env python3
"""Generate an offline human-review package for a six-keypoint YOLO pose model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KEYPOINT_NAMES = ("CR", "CL", "IR", "IL", "SR", "SL")
GT_COLOR = (35, 210, 95)
PRED_COLOR = (235, 65, 185)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


@dataclass(frozen=True)
class PoseLabel:
    keypoints: tuple[tuple[float, float, int], ...]


@dataclass(frozen=True)
class Prediction:
    keypoints: tuple[tuple[float, float, float], ...]
    box_confidence: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pose_label(path: Path) -> PoseLabel:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected one object, found {len(lines)}")
    values = lines[0].split()
    if len(values) != 23:
        raise ValueError(f"{path}: expected 23 fields, found {len(values)}")
    if int(float(values[0])) != 0:
        raise ValueError(f"{path}: expected class 0")
    numbers = [float(value) for value in values[5:]]
    keypoints = []
    for index in range(0, len(numbers), 3):
        x, y, visibility = numbers[index : index + 3]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{path}: keypoint coordinate outside [0,1]")
        keypoints.append((x, y, int(visibility)))
    if len(keypoints) != 6:
        raise ValueError(f"{path}: expected six keypoints")
    return PoseLabel(tuple(keypoints))


def find_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    missing = [path.name for path in images if path.stem not in labels]
    extra = sorted(set(labels) - {path.stem for path in images})
    if missing or extra:
        raise ValueError(f"image/label mismatch: missing_labels={missing[:5]}, extra_labels={extra[:5]}")
    return [(path, labels[path.stem]) for path in images]


def choose_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    if scale >= 1.0:
        return image.copy(), 1.0
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def draw_points(
    image: Image.Image,
    points: Sequence[tuple[float, float, float | int]],
    color: tuple[int, int, int],
    prefix: str,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = choose_font(max(16, round(min(output.size) * 0.018)))
    radius = max(5, round(min(output.size) * 0.007))
    for index, point in enumerate(points):
        x, y, third = point
        if third <= 0:
            continue
        px, py = round(x * output.width), round(y * output.height)
        draw.ellipse((px - radius - 2, py - radius - 2, px + radius + 2, py + radius + 2), fill=BLACK)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline=WHITE, width=1)
        label = f"{prefix}{KEYPOINT_NAMES[index]}"
        box = draw.textbbox((0, 0), label, font=font, stroke_width=2)
        tx = min(max(2, px + radius + 3), max(2, output.width - (box[2] - box[0]) - 4))
        ty = min(max(2, py - (box[3] - box[1]) // 2), max(2, output.height - (box[3] - box[1]) - 4))
        draw.text((tx, ty), label, font=font, fill=color, stroke_width=2, stroke_fill=BLACK)
    return output


def add_title(image: Image.Image, title: str, color: tuple[int, int, int]) -> Image.Image:
    font = choose_font(24)
    bar_height = 42
    canvas = Image.new("RGB", (image.width, image.height + bar_height), (25, 25, 25))
    canvas.paste(image, (0, bar_height))
    ImageDraw.Draw(canvas).text((12, 8), title, font=font, fill=color)
    return canvas


def calculate_sample_metrics(
    label: PoseLabel,
    prediction: Prediction,
    width: int,
    height: int,
) -> dict:
    diagonal = math.hypot(width, height)
    point_metrics = []
    for index, (gt_x, gt_y, visibility) in enumerate(label.keypoints):
        pred = prediction.keypoints[index] if index < len(prediction.keypoints) else (0.0, 0.0, 0.0)
        pred_x, pred_y, confidence = pred
        detected = confidence > 0 and (pred_x > 0 or pred_y > 0)
        distance_px = None
        distance_diagonal = None
        if visibility > 0 and detected:
            distance_px = math.hypot((pred_x - gt_x) * width, (pred_y - gt_y) * height)
            distance_diagonal = distance_px / diagonal
        point_metrics.append(
            {
                "name": KEYPOINT_NAMES[index],
                "visible": visibility > 0,
                "detected": detected,
                "confidence": round(float(confidence), 6),
                "distance_px": None if distance_px is None else round(distance_px, 3),
                "distance_diagonal": None if distance_diagonal is None else round(distance_diagonal, 6),
            }
        )
    distances = [item["distance_px"] for item in point_metrics if item["distance_px"] is not None]
    diagonal_distances = [item["distance_diagonal"] for item in point_metrics if item["distance_diagonal"] is not None]
    visible_count = sum(item["visible"] for item in point_metrics)
    detected_visible = sum(item["visible"] and item["detected"] for item in point_metrics)
    return {
        "points": point_metrics,
        "visible_count": visible_count,
        "detected_visible_count": detected_visible,
        "missing_count": visible_count - detected_visible,
        "mean_error_px": None if not distances else round(sum(distances) / len(distances), 3),
        "max_error_px": None if not distances else round(max(distances), 3),
        "mean_error_diagonal": None if not diagonal_distances else round(sum(diagonal_distances) / len(diagonal_distances), 6),
        "pck_20px_hits": sum(value <= 20 for value in distances),
        "pck_2pct_hits": sum(value <= 0.02 for value in diagonal_distances),
    }


def render_preview(
    image_path: Path,
    label: PoseLabel,
    prediction: Prediction,
    output_path: Path,
    max_panel_width: int = 900,
    max_panel_height: int = 1100,
) -> tuple[int, int]:
    with Image.open(image_path) as source:
        source.load()
        source = ImageOps.exif_transpose(source).convert("RGB")
        original_size = source.size
        base, _ = fit_image(source, max_panel_width, max_panel_height)
    truth = add_title(draw_points(base, label.keypoints, GT_COLOR, "GT-"), "Ground truth", GT_COLOR)
    predicted = add_title(draw_points(base, prediction.keypoints, PRED_COLOR, "P-"), "Prediction", PRED_COLOR)
    overlay = draw_points(base, label.keypoints, GT_COLOR, "GT-")
    overlay = add_title(draw_points(overlay, prediction.keypoints, PRED_COLOR, "P-"), "Overlay: green=GT, magenta=prediction", WHITE)
    gap = 8
    canvas = Image.new("RGB", (truth.width * 3 + gap * 2, truth.height), (70, 70, 70))
    canvas.paste(truth, (0, 0))
    canvas.paste(predicted, (truth.width + gap, 0))
    canvas.paste(overlay, ((truth.width + gap) * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=90, optimize=True)
    return original_size


def aggregate_metrics(samples: Sequence[dict]) -> dict:
    points = [point for sample in samples for point in sample["metrics"]["points"] if point["visible"]]
    measured = [point for point in points if point["distance_px"] is not None]
    errors_px = [point["distance_px"] for point in measured]
    errors_diag = [point["distance_diagonal"] for point in measured]
    sorted_errors = sorted(errors_px)

    def percentile(values: Sequence[float], fraction: float) -> float | None:
        if not values:
            return None
        return values[round((len(values) - 1) * fraction)]

    per_keypoint = {}
    for name in KEYPOINT_NAMES:
        named = [point for point in points if point["name"] == name]
        named_measured = [point for point in named if point["distance_px"] is not None]
        per_keypoint[name] = {
            "detected": len(named_measured),
            "total": len(named),
            "mean_error_px": None if not named_measured else round(sum(point["distance_px"] for point in named_measured) / len(named_measured), 3),
            "pck_20px": None if not named_measured else round(sum(point["distance_px"] <= 20 for point in named_measured) / len(named_measured), 6),
            "pck_2pct_diagonal": None if not named_measured else round(sum(point["distance_diagonal"] <= 0.02 for point in named_measured) / len(named_measured), 6),
        }
    return {
        "sample_count": len(samples),
        "visible_points": len(points),
        "detected_points": len(measured),
        "point_recall": None if not points else round(len(measured) / len(points), 6),
        "mean_error_px": None if not errors_px else round(sum(errors_px) / len(errors_px), 3),
        "median_error_px": percentile(sorted_errors, 0.5),
        "p90_error_px": percentile(sorted_errors, 0.9),
        "pck_20px": None if not measured else round(sum(value <= 20 for value in errors_px) / len(measured), 6),
        "pck_2pct_diagonal": None if not measured else round(sum(value <= 0.02 for value in errors_diag) / len(measured), 6),
        "samples_with_all_six": sum(sample["metrics"]["missing_count"] == 0 for sample in samples),
        "per_keypoint": per_keypoint,
    }


def write_index_csv(path: Path, samples: Sequence[dict]) -> None:
    fields = ["index", "filename", "box_confidence", "detected", "mean_error_px", "max_error_px", "mean_error_diagonal", "preview", "human_result", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples, 1):
            metrics = sample["metrics"]
            writer.writerow(
                {
                    "index": index,
                    "filename": sample["filename"],
                    "box_confidence": sample["box_confidence"],
                    "detected": f'{metrics["detected_visible_count"]}/{metrics["visible_count"]}',
                    "mean_error_px": metrics["mean_error_px"],
                    "max_error_px": metrics["max_error_px"],
                    "mean_error_diagonal": metrics["mean_error_diagonal"],
                    "preview": sample["preview"],
                    "human_result": "",
                    "notes": "",
                }
            )


HTML_TEMPLATE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>六点模型人工核验</title><style>
*{box-sizing:border-box}body{margin:0;background:#11151b;color:#edf2f7;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.top{position:sticky;top:0;z-index:3;background:#18202bcc;padding:12px 18px;backdrop-filter:blur(12px);border-bottom:1px solid #34404f}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,select,input,textarea{background:#202b38;color:#fff;border:1px solid #46566a;border-radius:7px;padding:7px 10px}button.active{border-color:#63b3ed;background:#174d72}.stats{color:#a9bad0;margin-top:7px}.main{max-width:1600px;margin:auto;padding:16px}.card{background:#1a222d;border:1px solid #34404f;border-radius:10px;padding:14px}.preview{width:100%;height:auto;display:block;background:#000}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px;margin:12px 0}.metric{background:#111821;border-radius:7px;padding:9px}.warn{color:#ffcc66}.bad{color:#ff8080}.good{color:#65e097}.actions button{font-size:16px}.notes{width:100%;min-height:62px;margin-top:10px}.foot{display:flex;justify-content:space-between;align-items:center;margin-top:12px}.hint{color:#95a6bb;font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:5px 7px;border-bottom:1px solid #34404f;text-align:right}td:first-child,th:first-child{text-align:left}</style></head><body>
<div class="top"><div class="row"><b>六点 Pose 模型人工核验包</b><select id="filter"><option value="all">全部</option><option value="todo">未核验</option><option value="bad">人工判为不准</option><option value="high">平均误差较大（≥20 px）</option><option value="miss">存在漏点</option></select><button id="prev">← 上一张</button><button id="next">下一张 →</button><input id="jump" type="number" min="1" style="width:90px"><button id="export">导出人工结果 CSV</button></div><div class="stats" id="stats"></div></div>
<main class="main"><section class="card"><h2 id="title"></h2><img class="preview" id="preview"><div class="meta" id="meta"></div><div id="pointTable"></div><div class="row actions"><b>人工判断：</b><button data-result="accurate">1 准确</button><button data-result="inaccurate">2 不准确</button><button data-result="unsure">3 不确定</button><button data-result="">清除</button></div><textarea class="notes" id="notes" placeholder="可记录哪一个点不准、偏向哪里等"></textarea><div class="foot"><span class="hint">快捷键：←/→ 切换，1准确，2不准确，3不确定。结果自动保存在本机浏览器。</span><span id="position"></span></div></section></main>
<script src="review_data.js"></script><script>(()=>{'use strict';const pkg=window.REVIEW_PACKAGE,samples=pkg.samples,key='six-point-model-review-'+pkg.package_id;let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}let filtered=[],pos=0;const $=id=>document.getElementById(id);const save=()=>localStorage.setItem(key,JSON.stringify(state));const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function refreshFilter(keep){const mode=$('filter').value,current=keep?filtered[pos]:null;filtered=samples.map((s,i)=>i).filter(i=>{const s=samples[i],r=(state[s.filename]||{}).result||'';if(mode==='todo')return !r;if(mode==='bad')return r==='inaccurate';if(mode==='high')return s.metrics.mean_error_px===null||s.metrics.mean_error_px>=20;if(mode==='miss')return s.metrics.missing_count>0;return true});if(current!==null&&filtered.includes(current))pos=filtered.indexOf(current);else pos=Math.min(pos,Math.max(0,filtered.length-1));render()}
function render(){const reviewed=Object.values(state).filter(v=>v.result).length;$('stats').textContent=`自动汇总：${pkg.summary.sample_count}张；点召回 ${(pkg.summary.point_recall*100).toFixed(1)}%；平均误差 ${pkg.summary.mean_error_px} px；PCK@20px ${(pkg.summary.pck_20px*100).toFixed(1)}%；人工已核验 ${reviewed}/${samples.length}`;if(!filtered.length){$('title').textContent='当前筛选无样本';$('preview').removeAttribute('src');$('meta').innerHTML='';$('pointTable').innerHTML='';$('position').textContent='0/0';return}const s=samples[filtered[pos]],m=s.metrics,entry=state[s.filename]||{};$('title').textContent=`${filtered[pos]+1}. ${s.filename}`;$('preview').src=s.preview;$('meta').innerHTML=`<div class="metric">检测点：<b class="${m.missing_count?'bad':'good'}">${m.detected_visible_count}/${m.visible_count}</b></div><div class="metric">平均误差：<b>${m.mean_error_px??'无法计算'} px</b></div><div class="metric">最大误差：<b>${m.max_error_px??'无法计算'} px</b></div><div class="metric">框置信度：<b>${Number(s.box_confidence).toFixed(4)}</b></div>`;$('pointTable').innerHTML='<table><tr><th>点</th><th>是否检测</th><th>点置信度</th><th>像素误差</th><th>图像对角线占比</th></tr>'+m.points.map(p=>`<tr><td>${p.name}</td><td>${p.detected?'是':'否'}</td><td>${Number(p.confidence).toFixed(4)}</td><td>${p.distance_px??'-'}</td><td>${p.distance_diagonal===null?'-':(p.distance_diagonal*100).toFixed(2)+'%'}</td></tr>`).join('')+'</table>';$('notes').value=entry.notes||'';document.querySelectorAll('[data-result]').forEach(b=>b.classList.toggle('active',b.dataset.result===(entry.result||'')));$('position').textContent=`筛选结果 ${pos+1}/${filtered.length}`;$('jump').value=filtered[pos]+1}
function setResult(result){if(!filtered.length)return;const s=samples[filtered[pos]];state[s.filename]={...(state[s.filename]||{}),result,notes:$('notes').value};save();if(result&&pos<filtered.length-1)pos++;refreshFilter(true)}function move(delta){if(!filtered.length)return;pos=Math.max(0,Math.min(filtered.length-1,pos+delta));render()}
document.querySelectorAll('[data-result]').forEach(b=>b.onclick=()=>setResult(b.dataset.result));$('notes').oninput=()=>{if(!filtered.length)return;const s=samples[filtered[pos]];state[s.filename]={...(state[s.filename]||{}),notes:$('notes').value};save()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('filter').onchange=()=>{pos=0;refreshFilter(false)};$('jump').onchange=()=>{const i=Number($('jump').value)-1;if(i>=0&&i<samples.length){$('filter').value='all';refreshFilter(false);pos=i;render()}};$('jump').onkeydown=e=>{if(e.key==='Enter')$('jump').dispatchEvent(new Event('change'))};
$('export').onclick=()=>{const cols=['index','filename','box_confidence','detected','mean_error_px','max_error_px','human_result','notes'];const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';const rows=[cols.join(',')];samples.forEach((s,i)=>{const e=state[s.filename]||{},m=s.metrics;rows.push([i+1,s.filename,s.box_confidence,`${m.detected_visible_count}/${m.visible_count}`,m.mean_error_px,m.max_error_px,e.result||'',e.notes||''].map(q).join(','))});const blob=new Blob(['\ufeff'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='六点模型人工核验结果.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(['1','2','3'].includes(e.key))setResult({1:'accurate',2:'inaccurate',3:'unsure'}[e.key])};refreshFilter(false)})();</script></body></html>"""


def write_readme(path: Path, model_path: Path, summary: dict) -> None:
    content = f"""# 最新六点 Pose 模型人工核验包

双击 `打开人工核验页面.html` 开始逐张核验。每张预览从左到右为人工真值、模型预测、叠加对比；绿色是人工真值，洋红色是模型预测。

- 模型：`{model_path}`
- 样本：患者隔离后的 test 集全量 {summary['sample_count']} 张
- 点召回：{summary['detected_points']}/{summary['visible_points']} = {summary['point_recall']:.2%}
- 平均像素误差：{summary['mean_error_px']} px
- 中位像素误差：{summary['median_error_px']} px
- 90 分位像素误差：{summary['p90_error_px']} px
- PCK@20 px：{summary['pck_20px']:.2%}
- PCK@图像对角线 2%：{summary['pck_2pct_diagonal']:.2%}
- 六点全部检出的图像：{summary['samples_with_all_six']}/{summary['sample_count']}

页面会把人工选择自动保存在当前浏览器，完成后点击“导出人工结果 CSV”。`样本索引与自动误差.csv` 是生成时的自动统计；`manifest.json` 保存完整逐点指标和文件哈希。

像素误差受原图分辨率影响，因此同时提供“图像对角线占比”。自动误差用于排序和筛查，最终是否临床可接受应以人工核验结果为准。
"""
    path.write_text(content, encoding="utf-8")


def build_package(
    image_dir: Path,
    label_dir: Path,
    model_path: Path,
    output_dir: Path,
    predictor: Callable[[Path], Prediction],
    overwrite: bool = False,
) -> dict:
    pairs = find_pairs(image_dir, label_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "previews").mkdir(parents=True)
    samples = []
    for index, (image_path, label_path) in enumerate(pairs, 1):
        label = parse_pose_label(label_path)
        prediction = predictor(image_path)
        preview_name = f"{index:04d}_{image_path.stem}.jpg"
        preview_path = output_dir / "previews" / preview_name
        width, height = render_preview(image_path, label, prediction, preview_path)
        metrics = calculate_sample_metrics(label, prediction, width, height)
        samples.append(
            {
                "filename": image_path.name,
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
                "preview": f"previews/{preview_name}",
                "width": width,
                "height": height,
                "box_confidence": round(prediction.box_confidence, 6),
                "metrics": metrics,
            }
        )
        print(f"[{index}/{len(pairs)}] {image_path.name}: detected={metrics['detected_visible_count']}/6 mean_error={metrics['mean_error_px']}", flush=True)
    samples.sort(key=lambda sample: (sample["metrics"]["mean_error_px"] is not None, sample["metrics"]["mean_error_px"] or float("inf")), reverse=True)
    summary = aggregate_metrics(samples)
    package_id = hashlib.sha256((sha256_file(model_path) + "|" + "|".join(sample["image_sha256"] for sample in samples)).encode()).hexdigest()[:16]
    manifest = {
        "package_id": package_id,
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "image_dir": str(image_dir.resolve()),
        "label_dir": str(label_dir.resolve()),
        "keypoint_names": KEYPOINT_NAMES,
        "summary": summary,
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    data = {"package_id": package_id, "summary": summary, "samples": samples}
    (output_dir / "review_data.js").write_text("window.REVIEW_PACKAGE=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    (output_dir / "打开人工核验页面.html").write_text(HTML_TEMPLATE, encoding="utf-8")
    write_index_csv(output_dir / "样本索引与自动误差.csv", samples)
    write_readme(output_dir / "README.md", model_path, summary)
    return manifest


def make_ultralytics_predictor(model_path: Path, confidence: float, image_size: int) -> Callable[[Path], Prediction]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    def predict(image_path: Path) -> Prediction:
        results = model.predict(str(image_path), conf=confidence, imgsz=image_size, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return Prediction(tuple((0.0, 0.0, 0.0) for _ in range(6)), 0.0)
        result = results[0]
        best = 0
        box_confidence = 0.0
        if result.boxes is not None and len(result.boxes):
            best = int(result.boxes.conf.argmax().item())
            box_confidence = float(result.boxes.conf[best].item())
        coordinates = result.keypoints.xyn[best].cpu().tolist()
        confidences = result.keypoints.conf
        if confidences is None:
            point_confidences = [1.0] * len(coordinates)
        else:
            point_confidences = confidences[best].cpu().tolist()
        keypoints = tuple(
            (float(coordinates[index][0]), float(coordinates[index][1]), float(point_confidences[index]))
            for index in range(min(6, len(coordinates)))
        )
        if len(keypoints) < 6:
            keypoints += tuple((0.0, 0.0, 0.0) for _ in range(6 - len(keypoints)))
        return Prediction(keypoints, box_confidence)

    return predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = make_ultralytics_predictor(args.model, args.conf, args.imgsz)
    manifest = build_package(args.image_dir, args.label_dir, args.model, args.output_dir, predictor, args.overwrite)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
