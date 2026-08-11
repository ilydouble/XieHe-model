#!/usr/bin/env python3
"""Render six-point annotations on an expanded canvas so off-image points remain visible."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

LABEL_SUFFIX = "_label.json"
POINT_LABELS = ("CL", "CR", "IL", "IR", "SL", "SR")
PAIR_LABELS = (("CL", "CR"), ("IL", "IR"), ("SL", "SR"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为归一化坐标越界的六点标注生成扩展画布复核图。")
    parser.add_argument("export_dir", type=Path, help="图像与 *_label.json 所在目录")
    parser.add_argument("--audit", required=True, type=Path, help="包含 coordinate_out_of_range 的审计JSON")
    parser.add_argument("--output-dir", required=True, type=Path, help="复核包输出目录")
    parser.add_argument(
        "--exclude-annotation",
        action="append",
        default=[],
        help="不生成的标注文件名，可重复传入",
    )
    parser.add_argument("--max-image-height", type=int, default=1800, help="复核图中原图最大显示高度")
    return parser.parse_args(argv)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def read_points(annotation_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    points: dict[str, dict[str, Any]] = {}
    for item in data.get("vertebrae", []):
        if not isinstance(item, dict) or item.get("label") not in POINT_LABELS:
            continue
        point = item.get("point")
        if not isinstance(point, dict):
            continue
        x, y = point.get("x"), point.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue
        points[str(item["label"])] = {
            "x": float(x),
            "y": float(y),
            "source": str(item.get("source", "unknown")),
        }
    return data, points


def point_is_out(point: dict[str, Any]) -> bool:
    return not (0 <= point["x"] <= 1 and 0 <= point["y"] <= 1)


def label_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font, stroke_width=1)
    padding = 5
    background = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )
    draw.rounded_rectangle(background, radius=5, fill=(0, 0, 0, 210), outline=fill, width=2)
    draw.text((x, y), text, font=font, fill=fill, stroke_width=1, stroke_fill=(0, 0, 0))


def render_one(
    image_path: Path,
    annotation_path: Path,
    issue_labels: set[str],
    output_path: Path,
    *,
    max_image_height: int,
) -> dict[str, Any]:
    data, points = read_points(annotation_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(1.5, max_image_height / height)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)

    x_values = [0.0, 1.0, *(point["x"] for point in points.values())]
    y_values = [0.0, 1.0, *(point["y"] for point in points.values())]
    margin = 0.07
    min_x = min(x_values) - margin
    max_x = max(x_values) + margin
    min_y = min(y_values) - margin
    max_y = max(y_values) + margin
    logical_width = max_x - min_x
    logical_height = max_y - min_y
    header_height = 150
    footer_height = 230
    canvas_width = max(900, round(display_width * logical_width))
    drawing_height = round(display_height * logical_height)
    canvas = Image.new("RGB", (canvas_width, header_height + drawing_height + footer_height), (35, 39, 47))
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = load_font(28)
    body_font = load_font(20)
    small_font = load_font(17)

    def to_canvas(x: float, y: float) -> tuple[float, float]:
        return (
            (x - min_x) * display_width,
            header_height + (y - min_y) * display_height,
        )

    image_x, image_y = to_canvas(0, 0)
    canvas.paste(image, (round(image_x), round(image_y)))
    boundary = (*to_canvas(0, 0), *to_canvas(1, 1))
    draw.rectangle(boundary, outline=(255, 255, 255, 255), width=4)
    draw.text((24, 18), annotation_path.name, font=title_font, fill=(255, 255, 255, 255))
    draw.text(
        (24, 62),
        f"原图尺寸 {width}×{height}｜白框=原图边界｜红色=越界",
        font=body_font,
        fill=(220, 225, 232, 255),
    )
    draw.text(
        (24, 98),
        "黄色=AI，青色=manual；灰色区域是为显示越界点扩展的画布",
        font=small_font,
        fill=(180, 188, 200, 255),
    )

    for left, right in PAIR_LABELS:
        if left in points and right in points:
            draw.line(
                (to_canvas(points[left]["x"], points[left]["y"]), to_canvas(points[right]["x"], points[right]["y"])),
                fill=(120, 160, 255, 180),
                width=5,
            )

    for label in POINT_LABELS:
        point = points.get(label)
        if point is None:
            continue
        x, y = to_canvas(point["x"], point["y"])
        is_out = point_is_out(point) or label in issue_labels
        color = (255, 70, 70) if is_out else (40, 225, 235) if point["source"] == "manual" else (255, 210, 45)
        radius = 15 if is_out else 11
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 220), outline=(255, 255, 255, 255), width=3)
        if is_out:
            draw.ellipse((x - radius - 9, y - radius - 9, x + radius + 9, y + radius + 9), outline=(*color, 230), width=4)
        text = f"{label} {point['source']} ({point['x']:.4f}, {point['y']:.4f})"
        text_bbox = draw.textbbox((0, 0), text, font=small_font, stroke_width=1)
        text_width = text_bbox[2] - text_bbox[0]
        label_x = x + 20
        if label_x + text_width + 16 > canvas_width:
            label_x = x - text_width - 30
        label_x = max(10, min(label_x, canvas_width - text_width - 16))
        label_box(
            draw,
            (label_x, y - 25),
            text,
            small_font,
            color,
        )

    footer_y = header_height + drawing_height + 18
    issue_lines = [
        f"越界点 {label}: x={points[label]['x']:.8f}, y={points[label]['y']:.8f}, source={points[label]['source']}"
        for label in sorted(issue_labels)
        if label in points
    ]
    for line_index, line in enumerate(issue_lines):
        draw.text(
            (24, footer_y + line_index * 34),
            line,
            font=body_font,
            fill=(255, 120, 120, 255),
        )
    note_y = footer_y + max(1, len(issue_lines)) * 34 + 12
    draw.text(
        (24, note_y),
        "注意：红点位于灰色扩展区时，表示真实保存坐标在原始图像范围之外。",
        font=small_font,
        fill=(225, 225, 225, 255),
    )
    original = str(data.get("originalFilename", ""))
    draw.text((24, note_y + 37), f"原始文件名：{original}", font=small_font, fill=(200, 205, 215, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return {
        "annotation": annotation_path.name,
        "image": image_path.name,
        "preview": output_path.name,
        "original_filename": original,
        "image_width": width,
        "image_height": height,
        "issue_labels": sorted(issue_labels),
        "points": points,
    }


def build_review_package(
    export_dir: Path,
    audit_path: Path,
    output_dir: Path,
    *,
    excluded_annotations: set[str] | None = None,
    max_image_height: int = 1800,
) -> dict[str, Any]:
    export_dir = export_dir.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    excluded = excluded_annotations or set()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    issues_by_file: defaultdict[str, set[str]] = defaultdict(set)
    for issue in audit.get("issues", []):
        if issue.get("code") != "coordinate_out_of_range":
            continue
        annotation = str(issue.get("file", ""))
        if annotation in excluded:
            continue
        message = str(issue.get("message", ""))
        label = message.split(".", 1)[0]
        if label in POINT_LABELS:
            issues_by_file[annotation].add(label)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for annotation_name in sorted(issues_by_file):
        annotation_path = export_dir / annotation_name
        stem = annotation_name[: -len(LABEL_SUFFIX)]
        image_path = export_dir / f"{stem}.png"
        if not annotation_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"缺少复核输入：{annotation_path} 或 {image_path}")
        preview_path = output_dir / f"{stem}_越界点扩展画布.png"
        records.append(
            render_one(
                image_path,
                annotation_path,
                issues_by_file[annotation_name],
                preview_path,
                max_image_height=max_image_height,
            )
        )

    index_path = output_dir / "越界点索引.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("标注文件", "图像文件", "预览文件", "越界标签", "来源", "x", "y", "像素x", "像素y")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for label in record["issue_labels"]:
                point = record["points"][label]
                writer.writerow(
                    {
                        "标注文件": record["annotation"],
                        "图像文件": record["image"],
                        "预览文件": record["preview"],
                        "越界标签": label,
                        "来源": point["source"],
                        "x": f"{point['x']:.12f}",
                        "y": f"{point['y']:.12f}",
                        "像素x": f"{point['x'] * record['image_width']:.3f}",
                        "像素y": f"{point['y'] * record['image_height']:.3f}",
                    }
                )

    cards = []
    for record in records:
        labels = ", ".join(record["issue_labels"])
        cards.append(
            "<article><h2>"
            + html.escape(record["annotation"])
            + "</h2><p>越界标签："
            + html.escape(labels)
            + "</p><img loading='lazy' src='"
            + html.escape(record["preview"])
            + "'></article>"
        )
    html_text = """<!doctype html><meta charset="utf-8"><title>越界点人工复核</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#171a20;color:#eee;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px}article{background:#252a33;padding:18px;border-radius:12px}img{width:100%;height:auto;background:#222}h1,h2{word-break:break-all}p{color:#ff9b9b}</style>
<h1>越界点人工复核</h1><p>白框为原始图像边界；红点为越界点；灰色区域是扩展画布。</p><main>""" + "".join(cards) + "</main>"
    (output_dir / "打开此文件人工复核.html").write_text(html_text, encoding="utf-8")

    summary = {
        "schema_version": 1,
        "export_dir": str(export_dir),
        "audit": str(audit_path),
        "excluded_annotations": sorted(excluded),
        "rendered_annotations": len(records),
        "rendered_out_of_range_points": sum(len(record["issue_labels"]) for record in records),
        "records": [
            {key: value for key, value in record.items() if key != "points"}
            for record in records
        ],
    }
    (output_dir / "复核清单.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_review_package(
        args.export_dir,
        args.audit,
        args.output_dir,
        excluded_annotations=set(args.exclude_annotation),
        max_image_height=args.max_image_height,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
