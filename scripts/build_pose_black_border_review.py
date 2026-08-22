#!/usr/bin/env python3
"""Build an offline visual-review package for suspicious black-border pose images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KEYPOINT_NAMES = ("CR", "CL", "IR", "IL", "SR", "SL")


@dataclass(frozen=True)
class BorderMetrics:
    width: int
    height: int
    left: int
    right: int
    top: int
    bottom: int
    border_area_fraction: float
    dark_pixel_fraction: float
    left_quarter_dark_fraction: float
    right_quarter_dark_fraction: float

    @property
    def aspect_wh(self) -> float:
        return self.width / self.height


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def parse_pose_label(path: Path) -> tuple[tuple[float, float, int], ...]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one object, found {len(lines)}")
    fields = lines[0].split()
    if len(fields) != 23:
        raise ValueError(f"{path}: expected 23 fields, found {len(fields)}")
    values = [float(value) for value in fields[5:]]
    points = []
    for offset in range(0, len(values), 3):
        x, y, visibility = values[offset : offset + 3]
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"{path}: keypoint outside [0,1]")
        points.append((x, y, int(visibility)))
    return tuple(points)


def continuous_dark_widths(gray: np.ndarray, threshold: int) -> tuple[int, int, int, int]:
    """Count consecutive outer rows/columns whose mean intensity is <= threshold."""
    row_dark = gray.mean(axis=1) <= threshold
    col_dark = gray.mean(axis=0) <= threshold

    def leading(values: np.ndarray) -> int:
        count = 0
        for value in values:
            if not value:
                break
            count += 1
        return count

    return leading(col_dark), leading(col_dark[::-1]), leading(row_dark), leading(row_dark[::-1])


def measure_image(image: Image.Image, threshold: int = 12) -> BorderMetrics:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    height, width = gray.shape
    left, right, top, bottom = continuous_dark_widths(gray, threshold)
    kept_w = max(0, width - left - right)
    kept_h = max(0, height - top - bottom)
    border_fraction = 1 - (kept_w * kept_h) / (width * height)
    quarter = max(1, width // 4)
    dark = gray <= threshold
    return BorderMetrics(
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        border_area_fraction=float(border_fraction),
        dark_pixel_fraction=float(dark.mean()),
        left_quarter_dark_fraction=float(dark[:, :quarter].mean()),
        right_quarter_dark_fraction=float(dark[:, -quarter:].mean()),
    )


def selection_reason(metrics: BorderMetrics, minimum_border_fraction: float = 0.05) -> str | None:
    if metrics.border_area_fraction >= minimum_border_fraction:
        return "continuous_ge_5pct"
    if (
        metrics.aspect_wh > 1.0
        and metrics.dark_pixel_fraction >= 0.40
        and metrics.left_quarter_dark_fraction >= 0.70
        and metrics.right_quarter_dark_fraction >= 0.70
    ):
        return "wide_interrupted_canvas"
    return None


def find_images(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"missing label for {image_path}")
            pairs.append((split, image_path, label_path))
    return pairs


def keypoints_in_border(
    points: Sequence[tuple[float, float, int]], metrics: BorderMetrics
) -> list[str]:
    risky = []
    for name, (x, y, visibility) in zip(KEYPOINT_NAMES, points):
        if visibility <= 0:
            continue
        px, py = x * metrics.width, y * metrics.height
        if px < metrics.left or px >= metrics.width - metrics.right or py < metrics.top or py >= metrics.height - metrics.bottom:
            risky.append(name)
    return risky


def fit_image(image: Image.Image, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    if scale == 1.0:
        return image.copy(), 1.0
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def render_preview(
    source: Image.Image,
    points: Sequence[tuple[float, float, int]],
    metrics: BorderMetrics,
    sample: dict,
    output_path: Path,
) -> None:
    image, scale = fit_image(source.convert("RGB"), 1080, 1120)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    left, right = round(metrics.left * scale), round(metrics.right * scale)
    top, bottom = round(metrics.top * scale), round(metrics.bottom * scale)
    w, h = image.size
    red = (255, 48, 48, 105)
    if left:
        draw_overlay.rectangle((0, 0, left - 1, h - 1), fill=red)
    if right:
        draw_overlay.rectangle((w - right, 0, w - 1, h - 1), fill=red)
    if top:
        draw_overlay.rectangle((0, 0, w - 1, top - 1), fill=red)
    if bottom:
        draw_overlay.rectangle((0, h - bottom, w - 1, h - 1), fill=red)
    keep_box = (left, top, max(left, w - right - 1), max(top, h - bottom - 1))
    draw_overlay.rectangle(keep_box, outline=(30, 235, 105, 255), width=3)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    point_font = choose_font(max(16, round(min(image.size) * 0.018)))
    radius = max(6, round(min(image.size) * 0.008))
    for name, (x, y, visibility) in zip(KEYPOINT_NAMES, points):
        if visibility <= 0:
            continue
        px, py = round(x * image.width), round(y * image.height)
        draw.ellipse((px - radius - 2, py - radius - 2, px + radius + 2, py + radius + 2), fill=(0, 0, 0))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(0, 225, 255), outline=(255, 255, 255), width=2)
        draw.text((px + radius + 4, py - radius), name, fill=(0, 225, 255), font=point_font, stroke_width=2, stroke_fill=(0, 0, 0))

    header_h, footer_h = 150, 150
    canvas = Image.new("RGB", (max(1120, image.width + 40), header_h + image.height + footer_h), (24, 26, 31))
    x0 = (canvas.width - image.width) // 2
    canvas.paste(image, (x0, header_h))
    title_font, body_font = choose_font(27), choose_font(20)
    cd = ImageDraw.Draw(canvas)
    warning = "关键点裁剪风险: " + ", ".join(sample["risky_keypoints"]) if sample["risky_keypoints"] else "关键点未落入统计暗边"
    reason_cn = "连续暗边面积≥5%" if sample["reason"] == "continuous_ge_5pct" else "横向黑画布（连续边算法漏检）"
    cd.text((20, 14), f'{sample["index"]:04d}  {sample["filename"]}', font=title_font, fill=(255, 255, 255))
    cd.text((20, 55), f'{sample["split"]} / {sample["source"]}  |  {reason_cn}', font=body_font, fill=(255, 190, 65))
    cd.text((20, 88), warning, font=body_font, fill=(255, 82, 82) if sample["risky_keypoints"] else (110, 235, 150))
    cd.text((20, 119), "红色=连续暗边统计区域；绿色框=拟保留区；青色=六个标注点（仅供核对，不代表可安全裁剪）", font=body_font, fill=(215, 220, 228))
    line1 = (
        f'{metrics.width}×{metrics.height}  W/H={metrics.aspect_wh:.3f}  '
        f'连续暗边面积={metrics.border_area_fraction:.2%}  全图暗像素={metrics.dark_pixel_fraction:.2%}'
    )
    line2 = (
        f'L/R/T/B={metrics.left}/{metrics.right}/{metrics.top}/{metrics.bottom}px  '
        f'= {metrics.left/metrics.width:.1%}/{metrics.right/metrics.width:.1%}/'
        f'{metrics.top/metrics.height:.1%}/{metrics.bottom/metrics.height:.1%}'
    )
    cd.text((20, header_h + image.height + 25), line1, font=body_font, fill=(245, 245, 245))
    cd.text((20, header_h + image.height + 65), line2, font=body_font, fill=(245, 245, 245))
    cd.text((20, header_h + image.height + 105), "注意：暗区均值阈值=12；文字、标尺或少量亮结构会打断连续边检测。", font=body_font, fill=(190, 198, 210))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=88, optimize=True)


def write_csv(path: Path, samples: Sequence[dict]) -> None:
    fields = [
        "index", "filename", "split", "source", "reason", "width", "height", "aspect_wh",
        "left_px", "right_px", "top_px", "bottom_px", "border_area_fraction",
        "dark_pixel_fraction", "left_quarter_dark_fraction", "right_quarter_dark_fraction",
        "risky_keypoints", "preview", "source_sha256", "preview_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {field: sample[field] for field in fields}
            row["risky_keypoints"] = ",".join(row["risky_keypoints"])
            writer.writerow(row)


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>六点Pose黑边可疑样本核验</title><style>
body{margin:0;background:#11151b;color:#eef2f7;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{position:sticky;top:0;z-index:3;background:#1c222c;padding:16px 22px;box-shadow:0 2px 10px #0008}h1{font-size:22px;margin:0 0 10px}.stats{color:#adb8c8;margin-bottom:10px}.controls{display:flex;flex-wrap:wrap;gap:8px}input,select{background:#0e1218;color:#fff;border:1px solid #3b4655;border-radius:6px;padding:8px 10px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;padding:18px}.card{background:#1c222c;border:1px solid #303a48;border-radius:9px;overflow:hidden}.card.risk{border-color:#ff5555}.card img{display:block;width:100%;height:420px;object-fit:contain;background:#080a0d}.meta{padding:10px 12px;line-height:1.45}.name{font-weight:650;word-break:break-all}.tag{display:inline-block;margin:5px 5px 0 0;padding:2px 7px;border-radius:10px;background:#354052;font-size:12px}.warn{background:#802d34}.empty{padding:40px;text-align:center;color:#aab4c2}a{color:inherit;text-decoration:none}
</style></head><body><header><h1>六点 Pose 黑边可疑样本核验</h1><div class="stats" id="stats"></div><div class="controls"><input id="q" placeholder="搜索文件名"><select id="split"><option value="">全部 split</option><option>train</option><option>val</option><option>test</option></select><select id="source"><option value="">全部来源</option><option>eap</option><option>old</option></select><select id="reason"><option value="">全部类型</option><option value="continuous_ge_5pct">连续暗边≥5%</option><option value="wide_interrupted_canvas">横向黑画布漏检</option></select><select id="risk"><option value="">全部风险</option><option value="yes">关键点裁剪风险</option><option value="no">无关键点风险</option></select><select id="sort"><option value="border">暗边面积从大到小</option><option value="name">文件名</option></select></div></header><main id="grid" class="grid"></main><script src="review_data.js"></script><script>
const $=id=>document.getElementById(id); const controls=['q','split','source','reason','risk','sort'];
function render(){let a=[...REVIEW_DATA.samples],q=$('q').value.trim().toLowerCase();a=a.filter(s=>(!q||s.filename.toLowerCase().includes(q))&&(!$('split').value||s.split===$('split').value)&&(!$('source').value||s.source===$('source').value)&&(!$('reason').value||s.reason===$('reason').value)&&(!$('risk').value||($('risk').value==='yes')===(s.risky_keypoints.length>0)));a.sort($('sort').value==='name'?(x,y)=>x.filename.localeCompare(y.filename):(x,y)=>y.border_area_fraction-x.border_area_fraction);$('stats').textContent=`显示 ${a.length} / ${REVIEW_DATA.summary.candidate_count} 张；红色为统计暗边，不等于安全裁剪区域。`;$('grid').innerHTML=a.length?a.map(s=>`<article class="card ${s.risky_keypoints.length?'risk':''}"><a href="${s.preview}" target="_blank"><img loading="lazy" src="${s.preview}"></a><div class="meta"><div class="name">${String(s.index).padStart(4,'0')} ${s.filename}</div><span class="tag">${s.split}/${s.source}</span><span class="tag">暗边 ${(s.border_area_fraction*100).toFixed(2)}%</span><span class="tag">${s.reason==='continuous_ge_5pct'?'连续暗边≥5%':'横向黑画布漏检'}</span>${s.risky_keypoints.length?`<span class="tag warn">裁剪风险 ${s.risky_keypoints.join(',')}</span>`:''}</div></article>`).join(''):'<div class="empty">没有符合当前筛选条件的样本</div>'}controls.forEach(id=>$(id).addEventListener(id==='q'?'input':'change',render));render();
</script></body></html>"""


def build_package(
    dataset_root: Path,
    output_dir: Path,
    threshold: int = 12,
    minimum_border_fraction: float = 0.05,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    previews = output_dir / "previews"
    previews.mkdir()
    selected = []
    for split, image_path, label_path in find_images(dataset_root):
        with Image.open(image_path) as opened:
            opened.load()
            source = ImageOps.exif_transpose(opened).copy()
        metrics = measure_image(source, threshold)
        reason = selection_reason(metrics, minimum_border_fraction)
        if reason is None:
            continue
        points = parse_pose_label(label_path)
        risky = keypoints_in_border(points, metrics)
        selected.append((metrics.border_area_fraction, split, image_path, source, points, metrics, reason, risky))
    selected.sort(key=lambda item: (-item[0], item[2].name))

    samples = []
    for index, (_, split, image_path, source, points, metrics, reason, risky) in enumerate(selected, 1):
        preview_rel = f"previews/{index:04d}_{image_path.stem}.jpg"
        sample = {
            "index": index,
            "filename": image_path.name,
            "split": split,
            "source": "eap" if image_path.name.startswith("eap_") else "old",
            "reason": reason,
            "width": metrics.width,
            "height": metrics.height,
            "aspect_wh": round(metrics.aspect_wh, 6),
            "left_px": metrics.left,
            "right_px": metrics.right,
            "top_px": metrics.top,
            "bottom_px": metrics.bottom,
            "border_area_fraction": round(metrics.border_area_fraction, 8),
            "dark_pixel_fraction": round(metrics.dark_pixel_fraction, 8),
            "left_quarter_dark_fraction": round(metrics.left_quarter_dark_fraction, 8),
            "right_quarter_dark_fraction": round(metrics.right_quarter_dark_fraction, 8),
            "risky_keypoints": risky,
            "preview": preview_rel,
            "source_sha256": sha256_file(image_path),
        }
        render_preview(source, points, metrics, sample, output_dir / preview_rel)
        sample["preview_sha256"] = sha256_file(output_dir / preview_rel)
        samples.append(sample)

    summary = {
        "dataset_image_count": len(find_images(dataset_root)),
        "candidate_count": len(samples),
        "threshold": threshold,
        "minimum_border_fraction": minimum_border_fraction,
        "by_split": dict(sorted(Counter(sample["split"] for sample in samples).items())),
        "by_source": dict(sorted(Counter(sample["source"] for sample in samples).items())),
        "by_reason": dict(sorted(Counter(sample["reason"] for sample in samples).items())),
        "keypoint_crop_risk_count": sum(bool(sample["risky_keypoints"]) for sample in samples),
    }
    manifest = {"schema_version": 1, "summary": summary, "samples": samples}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    (output_dir / "review_data.js").write_text(f"const REVIEW_DATA={data};\n", encoding="utf-8")
    (output_dir / "打开黑边核验页面.html").write_text(HTML, encoding="utf-8")
    write_csv(output_dir / "样本索引.csv", samples)
    readme = f"""# 六点 Pose 黑边可疑样本可视化包

- 数据集：`{dataset_root}`（只读扫描，未修改训练数据）
- 总图像：{summary['dataset_image_count']}
- 候选：{summary['candidate_count']}
- 连续暗边规则：灰度 8-bit，四边向内连续整行/整列平均亮度不超过 {threshold}，合计面积不少于 {minimum_border_fraction:.0%}
- 补充规则：W/H>1、全图暗像素≥40%、左右四分之一区暗像素均≥70%，用于发现被标尺/文字打断的横向黑画布
- 关键点裁剪风险样本：{summary['keypoint_crop_risk_count']}

双击 `打开黑边核验页面.html` 浏览、筛选并点开大图。预览中红色是统计暗边，绿色框是拟保留区，青色是标注点。此包只用于人工核对；连续暗边统计不能直接作为安全裁剪算法。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/pose_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=12)
    parser.add_argument("--minimum-border-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_package(args.dataset_root, args.output_dir, args.threshold, args.minimum_border_fraction)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
