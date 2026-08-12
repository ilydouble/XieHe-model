#!/usr/bin/env python3
"""Build an offline review package for two YOLO pose-corner label versions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


COLORS = {"current743": (0, 230, 255), "server": (255, 70, 140)}
VERTEBRAE = ("C7",) + tuple(f"T{i}" for i in range(1, 13)) + tuple(f"L{i}" for i in range(1, 6))


@dataclass(frozen=True)
class Pair:
    server_image: Path
    current_image: Path
    server_label: Path
    current_label: Path
    points_server: dict[int, list[tuple[float, float]]]
    points_current: dict[int, list[tuple[float, float]]]
    mean_delta: float
    max_delta: float
    per_class: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成服务器版与743版角点差异人工确认包")
    parser.add_argument("--candidates", required=True, type=Path, help="四列TSV：服务器图、当前图、相似度、来源")
    parser.add_argument("--server-root", required=True, type=Path, help="服务器数据根目录（含images/labels）")
    parser.add_argument("--current-root", required=True, type=Path, help="当前pose_corner_data根目录")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=30)
    return parser.parse_args()


def read_label(path: Path) -> dict[int, list[tuple[float, float]]]:
    result: dict[int, list[tuple[float, float]]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = line.split()
        if len(tokens) != 17:
            raise ValueError(f"{path}:{number} 应为17列，实际{len(tokens)}列")
        values = [float(token) for token in tokens]
        class_id = int(values[0])
        result[class_id] = [(values[5 + index * 3], values[6 + index * 3]) for index in range(4)]
    return result


def label_path(root: Path, image: Path) -> Path:
    return root / "labels" / image.parent.name / f"{image.stem}.txt"


def load_pairs(candidates: Path, server_root: Path, current_root: Path) -> list[Pair]:
    pairs: list[Pair] = []
    for line in candidates.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        server_raw, current_raw, *_ = line.split("\t")
        server_image, current_image = Path(server_raw), Path(current_raw)
        server_label = label_path(server_root, server_image)
        current_label = label_path(current_root, current_image)
        server_points, current_points = read_label(server_label), read_label(current_label)
        common = sorted((server_points.keys() & current_points.keys()) & set(range(1, 18)))
        if not common:
            raise ValueError(f"没有共同V1-V17类别：{server_image.name}")
        per_class: dict[int, float] = {}
        all_distances: list[float] = []
        for class_id in common:
            distances = [
                math.hypot(left[0] - right[0], left[1] - right[1])
                for left, right in zip(server_points[class_id], current_points[class_id], strict=True)
            ]
            per_class[class_id] = sum(distances) / len(distances)
            all_distances.extend(distances)
        pairs.append(Pair(
            server_image, current_image, server_label, current_label,
            server_points, current_points,
            sum(all_distances) / len(all_distances), max(all_distances), per_class,
        ))
    return pairs


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_version(draw: ImageDraw.ImageDraw, points: dict[int, list[tuple[float, float]]], width: int, height: int,
                 color: tuple[int, int, int], scale: float, classes: set[int] | None = None) -> None:
    radius = max(2, round(3 * scale))
    line_width = max(2, round(2 * scale))
    for class_id, normalized in sorted(points.items()):
        if class_id > 17 or (classes is not None and class_id not in classes):
            continue
        pixel = [(round(x * width), round(y * height)) for x, y in normalized]
        draw.line(pixel + [pixel[0]], fill=color, width=line_width)
        for x, y in pixel:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)


def crop_for_classes(image: Image.Image, pair: Pair, classes: list[int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    coords = []
    for class_id in classes:
        coords.extend(pair.points_server.get(class_id, []))
        coords.extend(pair.points_current.get(class_id, []))
    xs, ys = [x * width for x, _ in coords], [y * height for _, y in coords]
    margin_x = max(50, (max(xs) - min(xs)) * 0.30)
    margin_y = max(70, (max(ys) - min(ys)) * 0.35)
    box = (
        max(0, round(min(xs) - margin_x)), max(0, round(min(ys) - margin_y)),
        min(width, round(max(xs) + margin_x)), min(height, round(max(ys) + margin_y)),
    )
    return image.crop(box), box


def render_pair(pair: Pair, output: Path, rank: int) -> None:
    with Image.open(pair.current_image) as source:
        base = ImageOps.autocontrast(ImageOps.exif_transpose(source).convert("L"), cutoff=1).convert("RGB")
        original_width, original_height = source.size
    base.thumbnail((760, 1420), Image.Resampling.LANCZOS)
    full = base.copy()
    draw = ImageDraw.Draw(full)
    sx, sy = full.width / original_width, full.height / original_height
    # Coordinates are normalized, so use the rendered dimensions directly.
    draw_version(draw, pair.points_current, full.width, full.height, COLORS["current743"], min(sx, sy))
    draw_version(draw, pair.points_server, full.width, full.height, COLORS["server"], min(sx, sy))

    worst = sorted(pair.per_class, key=pair.per_class.get, reverse=True)[:6]
    with Image.open(pair.current_image) as source:
        original = ImageOps.autocontrast(ImageOps.exif_transpose(source).convert("L"), cutoff=1).convert("RGB")
    crops: list[tuple[Image.Image, str]] = []
    for class_id in worst:
        crop, box = crop_for_classes(original, pair, [class_id])
        crop.thumbnail((430, 220), Image.Resampling.LANCZOS)
        cd = ImageDraw.Draw(crop)
        x0, y0, x1, y1 = box
        cw, ch = x1 - x0, y1 - y0
        def shifted(points: dict[int, list[tuple[float, float]]], color: tuple[int, int, int]) -> None:
            normalized = {
                class_id: [((x * original.width - x0) / cw, (y * original.height - y0) / ch) for x, y in points[class_id]]
            }
            draw_version(cd, normalized, crop.width, crop.height, color, 1.0, {class_id})
        shifted(pair.points_current, COLORS["current743"])
        shifted(pair.points_server, COLORS["server"])
        crops.append((crop, f"{VERTEBRAE[class_id]}  平均差 {pair.per_class[class_id]:.4f}"))

    header_h, row_h, right_w = 130, 235, 450
    canvas = Image.new("RGB", (full.width + right_w, max(header_h + full.height, header_h + row_h * len(crops))), (22, 22, 22))
    canvas.paste(full, (0, header_h))
    d = ImageDraw.Draw(canvas)
    d.text((18, 12), f"#{rank:02d}  每图平均差 {pair.mean_delta:.4f}  最大差 {pair.max_delta:.4f}", fill="white", font=font(24))
    d.text((18, 50), "青色：743/NRRD版    洋红：服务器版", fill="white", font=font(22))
    d.text((18, 86), pair.server_image.name, fill=(210, 210, 210), font=font(15))
    for index, (crop, caption) in enumerate(crops):
        top = header_h + index * row_h
        canvas.paste(crop, (full.width + (right_w - crop.width) // 2, top + 30))
        d.text((full.width + 12, top + 4), caption, fill="white", font=font(17))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)


def write_html(output: Path, rows: list[dict[str, object]]) -> None:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html><meta charset=utf-8><title>角点双版本人工确认</title>
<style>body{font-family:-apple-system,sans-serif;background:#171717;color:#eee;margin:0}header{position:sticky;top:0;background:#222;padding:12px 20px;z-index:2}main{max-width:1300px;margin:auto;padding:18px}.card{display:none}.card.on{display:block}img{max-width:100%;height:auto}.buttons{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}button{font-size:18px;padding:10px 18px}small{color:#bbb}</style>
<header><b>服务器版 vs 743版高差异样本</b>　<span id=progress></span>　<button onclick=download()>导出CSV</button></header><main id=main></main>
<script>const rows=__DATA__;let at=0;const key='corner-review-choices-v1';const choices=JSON.parse(localStorage.getItem(key)||'{}');function show(){const r=rows[at];progress.textContent=`${at+1}/${rows.length} 已选择 ${Object.keys(choices).length} 当前：${choices[r.id]||'未选择'}`;main.innerHTML=`<section class=\"card on\"><h2>${r.id}　平均差 ${r.mean_delta}　最大差 ${r.max_delta}</h2><small>${r.server_image}<br>${r.current_image}</small><div class=buttons><button onclick=pick('server')>服务器版更准</button><button onclick=pick('current743')>743版更准</button><button onclick=pick('both_bad')>两版都不准</button><button onclick=pick('uncertain')>无法判断</button><button onclick=move(-1)>上一张</button><button onclick=move(1)>下一张</button></div><img src=\"${r.preview}\"></section>`}function pick(v){choices[rows[at].id]=v;localStorage.setItem(key,JSON.stringify(choices));if(at<rows.length-1)at++;show()}function move(n){at=Math.max(0,Math.min(rows.length-1,at+n));show()}function download(){let s='id,choice,server_image,current_image,mean_delta,max_delta\\n';for(const r of rows)s+=`${r.id},${choices[r.id]||''},${r.server_image},${r.current_image},${r.mean_delta},${r.max_delta}\\n`;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));a.download='人工确认结果.csv';a.click()}show()</script>""".replace("__DATA__", data)
    output.write_text(html, encoding="utf-8")


def build(candidates: Path, server_root: Path, current_root: Path, output_dir: Path, count: int) -> list[dict[str, object]]:
    pairs = sorted(load_pairs(candidates, server_root, current_root), key=lambda pair: pair.mean_delta, reverse=True)[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for rank, pair in enumerate(pairs, 1):
        preview = Path("previews") / f"{rank:02d}_{pair.server_image.stem}.jpg"
        render_pair(pair, output_dir / preview, rank)
        server_label_copy = Path("labels/server") / f"{rank:02d}_{pair.server_label.name}"
        current_label_copy = Path("labels/current743") / f"{rank:02d}_{pair.current_label.name}"
        (output_dir / server_label_copy).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / current_label_copy).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pair.server_label, output_dir / server_label_copy)
        shutil.copy2(pair.current_label, output_dir / current_label_copy)
        rows.append({
            "id": f"C{rank:02d}", "rank": rank, "mean_delta": round(pair.mean_delta, 6),
            "max_delta": round(pair.max_delta, 6), "server_image": pair.server_image.name,
            "current_image": pair.current_image.name, "preview": preview.as_posix(), "choice": "",
            "server_label": server_label_copy.as_posix(), "current743_label": current_label_copy.as_posix(),
            "worst_vertebrae": ";".join(VERTEBRAE[c] for c in sorted(pair.per_class, key=pair.per_class.get, reverse=True)[:6]),
        })
    fields = list(rows[0]) if rows else []
    with (output_dir / "样本索引与确认结果.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output_dir / "打开人工确认页面.html", rows)
    (output_dir / "README.md").write_text(
        "# 角点双版本高差异样本人工确认包\n\n"
        f"从全部重叠影像中按每图共同V1-V17角点平均距离降序抽取前{len(rows)}份。\n\n"
        "- 青色：当前743份的NRRD分割转角点版本。\n"
        "- 洋红：服务器seg_data转角点版本。\n"
        "- 右侧展示该图差异最大的6节椎体。\n"
        "- 打开HTML逐图选择；选择会保存在浏览器本地，结束后点击导出CSV。\n"
        "- labels目录含双方原始TXT标签副本；原始训练标签未修改。\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    args = parse_args()
    rows = build(args.candidates.resolve(), args.server_root.resolve(), args.current_root.resolve(), args.output_dir.resolve(), args.count)
    print(f"生成完成：{len(rows)}份 -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
