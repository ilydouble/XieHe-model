#!/usr/bin/env python3
"""Build a label-style visual comparison package: eap_ (new) vs legacy (old) six-point pose labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

KEYPOINT_NAMES = ("CR", "CL", "IR", "IL", "SR", "SL")
GROUPS = ("eap", "old")
GROUP_TITLE = {"eap": "新标注 eap_", "old": "旧数据"}
GROUP_COLOR = {"eap": (255, 90, 70), "old": (60, 170, 255)}
BANDS = (((0, 1), "锁骨 CR/CL"), ((2, 3), "髂骨 IR/IL"), ((4, 5), "骶骨 SR/SL"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count-per-group", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def parse_label(path: Path) -> list[tuple[float, float, int]]:
    values = path.read_text(encoding="utf-8").strip().split()
    numbers = [float(value) for value in values[5:]]
    return [(numbers[i], numbers[i + 1], int(numbers[i + 2])) for i in range(0, len(numbers), 3)]


def group_of(image_path: Path) -> str:
    return "eap" if image_path.name.startswith("eap_") else "old"


def collect_images(dataset_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for image in sorted((dataset_root / "images" / split).glob("*.png")):
            groups[group_of(image)].append(image)
    return groups


def deterministic_sample(images: list[Path], count: int) -> list[Path]:
    if count >= len(images):
        return images
    by_split: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        by_split[image.parent.name].append(image)
    raw = {split: count * len(group) / len(images) for split, group in by_split.items()}
    allocation = {split: int(value) for split, value in raw.items()}
    remaining = count - sum(allocation.values())
    for split in sorted(by_split, key=lambda name: (raw[name] - allocation[name], len(by_split[name]), name), reverse=True)[:remaining]:
        allocation[split] += 1
    selected: list[Path] = []
    for split in sorted(by_split):
        ordered = sorted(by_split[split], key=lambda path: hashlib.sha256(path.name.encode()).hexdigest())
        amount = allocation[split]
        if amount == 0:
            continue
        positions = [round(i * (len(ordered) - 1) / (amount - 1)) for i in range(amount)] if amount > 1 else [len(ordered) // 2]
        selected.extend(ordered[position] for position in positions)
    return selected


def draw_full(image: Image.Image, points: list[tuple[float, float, int]], color: tuple[int, int, int]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    text_font = font(max(14, round(canvas.height / 60)))
    radius = max(4, round(canvas.width / 160))
    for index, (x, y, visibility) in enumerate(points):
        if visibility <= 0:
            continue
        px, py = round(x * canvas.width), round(y * canvas.height)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=2)
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)
        draw.text((px + radius + 2, py - text_font.size // 2), KEYPOINT_NAMES[index], fill=color, font=text_font, stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas


def crop_band(original: Image.Image, points: list[tuple[float, float, int]], indices: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    coords = [(points[i][0] * original.width, points[i][1] * original.height) for i in indices]
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    margin_x, margin_y = max(60, (max(xs) - min(xs)) * 0.6), max(60, (max(ys) - min(ys)) * 0.6)
    box = (max(0, round(min(xs) - margin_x)), max(0, round(min(ys) - margin_y)), min(original.width, round(max(xs) + margin_x)), min(original.height, round(max(ys) + margin_y)))
    crop = original.crop(box)
    crop.thumbnail((520, 420), Image.Resampling.LANCZOS)
    x0, y0, x1, y1 = box
    shifted = [
        ((x * original.width - x0) / (x1 - x0), (y * original.height - y0) / (y1 - y0), v) if index in indices else (0.0, 0.0, 0)
        for index, (x, y, v) in enumerate(points)
    ]
    return draw_full(crop, shifted, color)


def render(image_path: Path, label_path: Path, output: Path, rank: int, group: str) -> None:
    points = parse_label(label_path)
    color = GROUP_COLOR[group]
    with Image.open(image_path) as source:
        original = ImageOps.autocontrast(ImageOps.exif_transpose(source).convert("L"), cutoff=1).convert("RGB")
    full = original.copy()
    full.thumbnail((760, 1450), Image.Resampling.LANCZOS)
    full = draw_full(full, points, color)
    crops = [(crop_band(original, points, indices, color), title) for indices, title in BANDS]
    header, right_w, row_h = 110, 560, 460
    canvas = Image.new("RGB", (full.width + right_w, max(header + full.height, header + row_h * len(crops))), (22, 22, 22))
    canvas.paste(full, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"#{rank:02d}  {GROUP_TITLE[group]}", fill=color, font=font(26))
    draw.text((18, 55), image_path.name, fill=(210, 210, 210), font=font(16))
    for index, (crop, title) in enumerate(crops):
        top = header + index * row_h
        draw.text((full.width + 15, top + 4), title, fill="white", font=font(22))
        canvas.paste(crop, (full.width + (right_w - crop.width) // 2, top + 38))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)


def write_html(path: Path, rows: list[dict[str, str]]) -> None:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = r"""<!doctype html><meta charset=utf-8><title>eap_ vs 旧数据标注风格抽样对照</title>
<style>body{font-family:-apple-system,sans-serif;background:#171717;color:#eee;margin:0}header{position:sticky;top:0;background:#222;padding:12px 20px;z-index:2}main{max-width:1400px;margin:auto;padding:18px}img{max-width:100%}button{font-size:16px;padding:9px 16px;margin:6px}select{font-size:16px;padding:6px}</style>
<header><b>eap_ 新标注 vs 旧数据　标注风格抽样对照</b>　<select id=filter><option value=all>全部</option><option value=eap>仅看 eap_</option><option value=old>仅看旧数据</option></select>　<span id=progress></span>　<button onclick=download()>导出CSV</button></header>
<main id=main></main>
<script>const rows=__DATA__;let view=rows.map((r,i)=>i);let at=0;const key='pose-style-review-v1';const choices=JSON.parse(localStorage.getItem(key)||'{}');
function applyFilter(){const mode=filter.value;view=rows.map((r,i)=>i).filter(i=>mode==='all'||rows[i].group===mode);at=0;show()}
function show(){if(!view.length){main.innerHTML='<p>无样本</p>';progress.textContent='0/0';return}const idx=view[at],r=rows[idx];progress.textContent=`${at+1}/${view.length}（${r.group}）　已标记 ${Object.keys(choices).length}　当前：${choices[r.id]||'未标记'}`;main.innerHTML=`<h2>${r.id}　${r.group_title}</h2><small>${r.image}</small><div><button onclick=pick('typical')>典型/符合本组习惯</button><button onclick=pick('offset')>取点位置明显偏移</button><button onclick=pick('uncertain')>无法判断</button><button onclick=move(-1)>上一张</button><button onclick=move(1)>下一张</button></div><img src="${r.preview}">`}
function pick(v){choices[rows[view[at]].id]=v;localStorage.setItem(key,JSON.stringify(choices));if(at<view.length-1)at++;show()}
function move(n){at=Math.max(0,Math.min(view.length-1,at+n));show()}
function download(){let s='id,choice,group,image\n';for(const r of rows)s+=`${r.id},${choices[r.id]||''},${r.group},${r.image}\n`;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));a.download='标注风格核对结果.csv';a.click()}
filter.onchange=applyFilter;applyFilter();</script>""".replace("__DATA__", data)
    path.write_text(html, encoding="utf-8")


def build(dataset_root: Path, output_dir: Path, count_per_group: int, overwrite: bool) -> list[dict[str, str]]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    groups = collect_images(dataset_root)
    rows: list[dict[str, str]] = []
    rank_counter = {"eap": 0, "old": 0}
    for group in GROUPS:
        selected = deterministic_sample(groups[group], min(count_per_group, len(groups[group])))
        for image in selected:
            rank_counter[group] += 1
            rank = rank_counter[group]
            label = dataset_root / "labels" / image.parent.name / f"{image.stem}.txt"
            preview = Path("previews") / group / f"{rank:02d}_{image.stem}.jpg"
            render(image, label, output_dir / preview, rank, group)
            copied_label = Path("labels") / group / f"{rank:02d}_{label.name}"
            (output_dir / copied_label).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, output_dir / copied_label)
            rows.append({
                "id": f"{group.upper()}{rank:02d}", "group": group, "group_title": GROUP_TITLE[group],
                "split": image.parent.name, "image": image.name, "label": copied_label.as_posix(),
                "preview": preview.as_posix(), "choice": "",
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (output_dir / "样本索引.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps({
        "eap_total": len(groups["eap"]), "old_total": len(groups["old"]),
        "eap_sampled": rank_counter["eap"], "old_sampled": rank_counter["old"], "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output_dir / "打开人工对照页面.html", rows)
    (output_dir / "README.md").write_text(
        "# eap_ 新标注 vs 旧数据 标注风格抽样对照包\n\n"
        f"数据来源：`{dataset_root}`（train/val/test 全量按比例分层抽样，稳定哈希排序，非按误差筛选）。\n\n"
        f"- eap_（新标注）共 {len(groups['eap'])} 份，抽样 {rank_counter['eap']} 份\n"
        f"- 旧数据共 {len(groups['old'])} 份，抽样 {rank_counter['old']} 份\n\n"
        "每张左侧为整图六点标注（橙色=eap_，蓝色=旧数据），右侧从上到下放大锁骨(CR/CL)、髂骨(IR/IL)、骶骨(SR/SL)三个区域，"
        "便于对比两批标注在同一骨性标志上具体取点位置的习惯差异。\n\n"
        "打开 `打开人工对照页面.html` 浏览，可按分组筛选，也可标记“典型/偏移/无法判断”，结果保存在浏览器本地，点击导出CSV。\n"
        "`labels/` 目录内是对应标签副本；原始训练数据未修改。\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    args = parse_args()
    rows = build(args.dataset_root.resolve(), args.output_dir.resolve(), args.count_per_group, args.overwrite)
    print(f"生成完成：eap_ {sum(r['group']=='eap' for r in rows)} 份，旧数据 {sum(r['group']=='old' for r in rows)} 份 -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
