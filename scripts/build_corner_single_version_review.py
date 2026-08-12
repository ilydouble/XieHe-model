#!/usr/bin/env python3
"""Build an offline review package for sampled single-version pose-corner labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


VERTEBRAE = ("C7",) + tuple(f"T{i}" for i in range(1, 13)) + tuple(f"L{i}" for i in range(1, 6))
COLORS = ((255, 90, 70), (255, 190, 40), (80, 220, 255), (60, 245, 150))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成未与服务器重叠的743版角点抽样确认包")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--matched-candidates", required=True, type=Path, help="服务器-当前同图候选TSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=30)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_label(path: Path) -> dict[int, list[tuple[float, float]]]:
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = [float(token) for token in line.split()]
        if len(values) != 17:
            raise ValueError(f"{path}:{number}不是17列")
        result[int(values[0])] = [(values[5 + 3 * index], values[6 + 3 * index]) for index in range(4)]
    return result


def remaining_images(dataset_root: Path, matched_candidates: Path) -> list[Path]:
    matched = {
        Path(line.split("\t")[1]).resolve()
        for line in matched_candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return sorted(
        (path for path in (dataset_root / "images").glob("*/*.png") if not path.name.startswith("eap_") and path.resolve() not in matched),
        key=lambda path: (path.parent.name, path.name),
    )


def deterministic_stratified_sample(images: list[Path], count: int) -> list[Path]:
    if count >= len(images):
        return images
    ordered = sorted(images, key=lambda path: hashlib.sha256(path.name.encode()).hexdigest())
    # Evenly span a stable hash order so the sample is reproducible and not biased by filename/date.
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)] if count > 1 else [0]
    return [ordered[position] for position in positions]


def draw_points(draw: ImageDraw.ImageDraw, points: dict[int, list[tuple[float, float]]], width: int, height: int, *, labels: bool) -> None:
    text_font = font(max(12, round(height / 150)))
    line_width = max(2, round(width / 500))
    for class_id, normalized in sorted(points.items()):
        if class_id > 17:
            continue
        color = COLORS[class_id % len(COLORS)]
        pixel = [(round(x * width), round(y * height)) for x, y in normalized]
        draw.line(pixel + [pixel[0]], fill=color, width=line_width)
        if labels:
            x, y = pixel[0]
            name = VERTEBRAE[class_id] if class_id < len(VERTEBRAE) else f"V{class_id}"
            draw.rectangle((x - 2, y - 2, x + text_font.size * 2, y + text_font.size + 3), fill=(0, 0, 0))
            draw.text((x, y), name, fill="white", font=text_font)


def crop_band(original: Image.Image, points: dict[int, list[tuple[float, float]]], classes: range) -> Image.Image:
    selected = [xy for class_id in classes for xy in points.get(class_id, [])]
    xs = [x * original.width for x, _ in selected]; ys = [y * original.height for _, y in selected]
    margin_x = max(80, (max(xs) - min(xs)) * 0.25); margin_y = max(80, (max(ys) - min(ys)) * 0.12)
    box = (max(0, round(min(xs) - margin_x)), max(0, round(min(ys) - margin_y)), min(original.width, round(max(xs) + margin_x)), min(original.height, round(max(ys) + margin_y)))
    crop = original.crop(box)
    crop.thumbnail((620, 550), Image.Resampling.LANCZOS)
    shifted = {}
    x0, y0, x1, y1 = box
    for class_id in classes:
        if class_id in points:
            shifted[class_id] = [((x * original.width - x0) / (x1 - x0), (y * original.height - y0) / (y1 - y0)) for x, y in points[class_id]]
    draw_points(ImageDraw.Draw(crop), shifted, crop.width, crop.height, labels=True)
    return crop


def render(image_path: Path, label_path: Path, output: Path, rank: int) -> None:
    points = read_label(label_path)
    with Image.open(image_path) as source:
        original = ImageOps.autocontrast(ImageOps.exif_transpose(source).convert("L"), cutoff=1).convert("RGB")
    full = original.copy(); full.thumbnail((760, 1450), Image.Resampling.LANCZOS)
    draw_points(ImageDraw.Draw(full), points, full.width, full.height, labels=True)
    bands = [(range(1, 7), "T1–T6"), (range(7, 13), "T7–T12"), (range(13, 18), "L1–L5")]
    crops = [(crop_band(original, points, classes), title) for classes, title in bands]
    header, right_w = 110, 650
    canvas = Image.new("RGB", (full.width + right_w, max(header + full.height, header + 590 * len(crops))), (22, 22, 22))
    canvas.paste(full, (0, header)); d = ImageDraw.Draw(canvas)
    d.text((18, 12), f"#{rank:02d}  743/NRRD版未重叠样本", fill="white", font=font(26))
    d.text((18, 55), image_path.name, fill=(210, 210, 210), font=font(16))
    for index, (crop, title) in enumerate(crops):
        top = header + index * 590
        d.text((full.width + 15, top + 4), title, fill="white", font=font(22))
        canvas.paste(crop, (full.width + (right_w - crop.width) // 2, top + 38))
    output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output, quality=92, optimize=True)


def write_html(path: Path, rows: list[dict[str, str]]) -> None:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html><meta charset=utf-8><title>743版未重叠角点抽样确认</title><style>body{font-family:-apple-system,sans-serif;background:#171717;color:#eee;margin:0}header{position:sticky;top:0;background:#222;padding:12px 20px;z-index:2}main{max-width:1400px;margin:auto;padding:18px}img{max-width:100%}button{font-size:18px;padding:10px 18px;margin:8px}</style><header><b>743版未重叠角点抽样确认</b>　<span id=progress></span>　<button onclick=download()>导出CSV</button></header><main id=main></main><script>const rows=__DATA__;let at=0;const key='corner-single-review-v1';const choices=JSON.parse(localStorage.getItem(key)||'{}');function show(){const r=rows[at];progress.textContent=`${at+1}/${rows.length} 已选择 ${Object.keys(choices).length} 当前：${choices[r.id]||'未选择'}`;main.innerHTML=`<h2>${r.id} ${r.split}</h2><small>${r.image}</small><div><button onclick=pick('correct')>整体正确</button><button onclick=pick('wrong_numbering')>椎体编号错位</button><button onclick=pick('wrong_corners')>角点不准</button><button onclick=pick('both_wrong')>编号和角点都不对</button><button onclick=pick('uncertain')>无法判断</button><button onclick=move(-1)>上一张</button><button onclick=move(1)>下一张</button></div><img src=\"${r.preview}\">`};function pick(v){choices[rows[at].id]=v;localStorage.setItem(key,JSON.stringify(choices));if(at<rows.length-1)at++;show()}function move(n){at=Math.max(0,Math.min(rows.length-1,at+n));show()}function download(){let s='id,choice,split,image,label\\n';for(const r of rows)s+=`${r.id},${choices[r.id]||''},${r.split},${r.image},${r.label}\\n`;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));a.download='人工确认结果.csv';a.click()}show()</script>""".replace("__DATA__", data)
    path.write_text(html, encoding="utf-8")


def build(dataset_root: Path, matched_candidates: Path, output_dir: Path, count: int) -> list[dict[str, str]]:
    all_remaining = remaining_images(dataset_root, matched_candidates)
    selected = deterministic_stratified_sample(all_remaining, min(count, len(all_remaining)))
    output_dir.mkdir(parents=True, exist_ok=True); rows = []
    for rank, image in enumerate(selected, 1):
        label = dataset_root / "labels" / image.parent.name / f"{image.stem}.txt"
        preview = Path("previews") / f"{rank:02d}_{image.stem}.jpg"; render(image, label, output_dir / preview, rank)
        copied_label = Path("labels") / f"{rank:02d}_{label.name}"; (output_dir / copied_label).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(label, output_dir / copied_label)
        rows.append({"id": f"S{rank:02d}", "split": image.parent.name, "image": image.name, "label": copied_label.as_posix(), "preview": preview.as_posix(), "choice": ""})
    fields = list(rows[0]) if rows else []
    with (output_dir / "样本索引与确认结果.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps({"remaining_count": len(all_remaining), "sample_count": len(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output_dir / "打开人工确认页面.html", rows)
    (output_dir / "README.md").write_text(f"# 743版未重叠角点抽样确认包\n\n743份排除与服务器同影像的490份后剩余{len(all_remaining)}份。本包按稳定哈希顺序均匀抽取{len(rows)}份，覆盖不同文件名与检查，非按标注质量筛选。\n\n整图标出V1-V17，右侧分别放大T1-T6、T7-T12、L1-L5。打开HTML选择整体正确、编号错位、角点不准、两者都不对或无法判断。选择自动保存在浏览器本地。原数据未修改。\n", encoding="utf-8")
    return rows


def main() -> None:
    args = parse_args(); rows = build(args.dataset_root.resolve(), args.matched_candidates.resolve(), args.output_dir.resolve(), args.count)
    print(f"生成完成：{len(rows)}份 -> {args.output_dir.resolve()}")


if __name__ == "__main__": main()
