#!/usr/bin/env python3
"""Build a before/after review package for the 4 manually-confirmed horizontal-flip
corrections applied to the E-drive training export.

For each of the 4 flipped stems, shows the original (pre-flip) image+labels next to
the corrected (post-flip) image+labels, both annotated with the six spine points
(CL/CR/IL/IR/SL/SR), so the flip + label swap can be visually verified.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

POINT_LABELS = ("CL", "CR", "IL", "IR", "SL", "SR")
POINT_COLOR = {
    "CL": (255, 90, 70), "CR": (255, 160, 60),
    "IL": (90, 200, 255), "IR": (60, 130, 255),
    "SL": (140, 255, 120), "SR": (60, 200, 90),
}

FLIP_STEMS = [
    "158_正位Xray",
    "165_e2629bb9-4edc-4d7a-8956-e9aeece8eb1a",
    "2115_SCO2105P0022_20210514",
    "4500_截屏2026-07-23 10.22.16",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", required=True, type=Path, help="翻转前原图/JSON备份目录")
    parser.add_argument("--after-dir", required=True, type=Path, help="翻转后当前训练导出目录")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_points(label_path: Path) -> dict[str, tuple[float, float]]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    points: dict[str, tuple[float, float]] = {}
    for item in data.get("vertebrae", []):
        if item.get("label") in POINT_LABELS and isinstance(item.get("point"), dict):
            points[item["label"]] = (float(item["point"]["x"]), float(item["point"]["y"]))
    return points


def draw_points(image: Image.Image, points: dict[str, tuple[float, float]]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    text_font = font(max(14, round(canvas.height / 55)))
    radius = max(4, round(canvas.width / 150))
    for label, (x, y) in points.items():
        color = POINT_COLOR[label]
        px, py = round(x * canvas.width), round(y * canvas.height)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=3)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
        draw.text((px + radius + 3, py - text_font.size // 2), label, fill=color, font=text_font, stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas


def load_panel(image_path: Path, label_path: Path) -> tuple[Image.Image, dict[str, tuple[float, float]]]:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return image, read_points(label_path)


def render(stem: str, before_dir: Path, after_dir: Path, output: Path, rank: int) -> dict[str, object]:
    before_image, before_points = load_panel(before_dir / f"{stem}.png", before_dir / f"{stem}_label.json")
    after_image, after_points = load_panel(after_dir / f"{stem}.png", after_dir / f"{stem}_label.json")

    panel_max = (620, 1350)
    before_thumb = before_image.copy()
    before_thumb.thumbnail(panel_max, Image.Resampling.LANCZOS)
    before_thumb = draw_points(before_thumb, before_points)
    after_thumb = after_image.copy()
    after_thumb.thumbnail(panel_max, Image.Resampling.LANCZOS)
    after_thumb = draw_points(after_thumb, after_points)

    header, gap = 90, 30
    width = before_thumb.width + after_thumb.width + gap
    height = header + max(before_thumb.height, after_thumb.height)
    canvas = Image.new("RGB", (width, height), (22, 22, 22))
    canvas.paste(before_thumb, (0, header))
    canvas.paste(after_thumb, (before_thumb.width + gap, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"#{rank:02d}  {stem}", fill="white", font=font(24))
    draw.text((18, 48), "左：翻转前（原始保存方向）　右：翻转后（当前训练目录，已镜像+坐标x'=1-x）", fill=(210, 210, 210), font=font(16))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)
    return {"before_points": {k: list(v) for k, v in before_points.items()}, "after_points": {k: list(v) for k, v in after_points.items()}}


def write_html(path: Path, rows: list[dict[str, str]]) -> None:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = r"""<!doctype html><meta charset=utf-8><title>水平翻转修正人工复核</title>
<style>body{font-family:-apple-system,sans-serif;background:#171717;color:#eee;margin:0}header{position:sticky;top:0;background:#222;padding:12px 20px;z-index:2}main{max-width:1400px;margin:auto;padding:18px}img{max-width:100%}button{font-size:16px;padding:9px 16px;margin:6px}</style>
<header><b>4 张水平翻转修正样本人工复核</b>　<span id=progress></span>　<button onclick=download()>导出CSV</button></header>
<main id=main></main>
<script>const rows=__DATA__;let at=0;const key='pose-flip-review-v1';const choices=JSON.parse(localStorage.getItem(key)||'{}');
function show(){const r=rows[at];progress.textContent=`${at+1}/${rows.length}　已标记 ${Object.keys(choices).length}　当前：${choices[r.id]||'未标记'}`;main.innerHTML=`<h2>${r.id}　${r.stem}</h2><div><button onclick=pick('correct')>翻转+标签正确</button><button onclick=pick('wrong')>有问题</button><button onclick=pick('uncertain')>无法判断</button><button onclick=move(-1)>上一张</button><button onclick=move(1)>下一张</button></div><img src="${r.preview}">`}
function pick(v){choices[rows[at].id]=v;localStorage.setItem(key,JSON.stringify(choices));if(at<rows.length-1)at++;show()}
function move(n){at=Math.max(0,Math.min(rows.length-1,at+n));show()}
function download(){let s='id,choice,stem\n';for(const r of rows)s+=`${r.id},${choices[r.id]||''},${r.stem}\n`;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));a.download='翻转修正核对结果.csv';a.click()}
show()</script>""".replace("__DATA__", data)
    path.write_text(html, encoding="utf-8")


def build(before_dir: Path, after_dir: Path, output_dir: Path, overwrite: bool) -> list[dict[str, object]]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for rank, stem in enumerate(FLIP_STEMS, 1):
        preview = Path("previews") / f"{rank:02d}_{stem}.jpg"
        points = render(stem, before_dir, after_dir, output_dir / preview, rank)
        for tag, src_dir in (("before", before_dir), ("after", after_dir)):
            dst = output_dir / "labels" / tag / f"{rank:02d}_{stem}_label.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dir / f"{stem}_label.json", dst)
        rows.append({
            "id": f"F{rank:02d}", "stem": stem, "preview": preview.as_posix(), "choice": "",
            **points,
        })

    fields = ["id", "stem", "preview", "choice"]
    with (output_dir / "样本索引.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    (output_dir / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output_dir / "打开人工复核页面.html", rows)
    (output_dir / "README.md").write_text(
        "# 水平翻转修正样本人工复核包\n\n"
        f"覆盖此前人工确认应做水平翻转的 4 张六点标注图（{', '.join(FLIP_STEMS)}）。\n\n"
        "每张左侧为翻转前（原始保存方向与坐标），右侧为翻转后（当前训练导出目录中的实际内容：像素镜像 + 坐标 x'=1-x + CL/CR、IL/IR、SL/SR 身份互换）。\n\n"
        "打开 `打开人工复核页面.html` 逐张核对翻转与标签交换是否正确；结果保存在浏览器本地，可导出CSV。\n"
        "`labels/before` 与 `labels/after` 为对应JSON标签副本；原始训练数据未被本次生成脚本修改。\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    args = parse_args()
    rows = build(args.before_dir.resolve(), args.after_dir.resolve(), args.output_dir.resolve(), args.overwrite)
    print(f"生成完成：{len(rows)} 份 -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
