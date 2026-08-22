#!/usr/bin/env python3
"""Build deterministic ROI crop views for mixed full-image/ROI YOLO Pose training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PoseLabel:
    class_id: int
    bbox: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, int], ...]


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


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
    fields = lines[0].split()
    if len(fields) != 23:
        raise ValueError(f"{path}: expected 23 fields, found {len(fields)}")
    class_value = float(fields[0])
    if not class_value.is_integer():
        raise ValueError(f"{path}: class id must be an integer")
    bbox = tuple(float(value) for value in fields[1:5])
    if any(not 0.0 <= value <= 1.0 for value in bbox):
        raise ValueError(f"{path}: bbox field outside [0,1]")
    keypoint_values = [float(value) for value in fields[5:]]
    keypoints = []
    for offset in range(0, len(keypoint_values), 3):
        x, y, visibility = keypoint_values[offset : offset + 3]
        if visibility not in (0.0, 1.0, 2.0):
            raise ValueError(f"{path}: invalid visibility {visibility}")
        if visibility > 0 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{path}: visible keypoint outside [0,1]")
        keypoints.append((x, y, int(visibility)))
    if len(keypoints) != 6:
        raise ValueError(f"{path}: expected six keypoints")
    return PoseLabel(int(class_value), bbox, tuple(keypoints))


def format_pose_label(label: PoseLabel) -> str:
    fields = [str(label.class_id), *(f"{value:.8f}" for value in label.bbox)]
    for x, y, visibility in label.keypoints:
        fields.extend((f"{x:.8f}", f"{y:.8f}", str(visibility)))
    return " ".join(fields) + "\n"


def target_bounds(label: PoseLabel, width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, bbox_w, bbox_h = label.bbox
    left = (cx - bbox_w / 2) * width
    right = (cx + bbox_w / 2) * width
    top = (cy - bbox_h / 2) * height
    bottom = (cy + bbox_h / 2) * height
    visible = [(x * width, y * height) for x, y, visibility in label.keypoints if visibility > 0]
    if not visible:
        raise ValueError("label has no visible keypoints")
    left = min(left, *(x for x, _ in visible))
    right = max(right, *(x for x, _ in visible))
    top = min(top, *(y for _, y in visible))
    bottom = max(bottom, *(y for _, y in visible))
    return max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom)


def deterministic_rng(filename: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{filename}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def compute_crop_box(
    label: PoseLabel,
    width: int,
    height: int,
    filename: str,
    seed: int = 20260822,
    margin: float = 0.20,
    shift_jitter: float = 0.05,
    scale_jitter: float = 0.10,
    safety_margin: float = 0.03,
    minimum_size: int = 64,
) -> CropBox:
    if width < 2 or height < 2:
        raise ValueError(f"invalid image size {width}x{height}")
    target_left, target_top, target_right, target_bottom = target_bounds(label, width, height)
    target_w = max(1.0, target_right - target_left)
    target_h = max(1.0, target_bottom - target_top)
    rng = deterministic_rng(filename, seed)
    scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    crop_w = max(float(minimum_size), target_w * (1.0 + 2.0 * margin) * scale)
    crop_h = max(float(minimum_size), target_h * (1.0 + 2.0 * margin) * scale)
    center_x = (target_left + target_right) / 2 + rng.uniform(-shift_jitter, shift_jitter) * target_w
    center_y = (target_top + target_bottom) / 2 + rng.uniform(-shift_jitter, shift_jitter) * target_h
    left, right = center_x - crop_w / 2, center_x + crop_w / 2
    top, bottom = center_y - crop_h / 2, center_y + crop_h / 2

    guard_x, guard_y = safety_margin * target_w, safety_margin * target_h
    left = min(left, target_left - guard_x)
    right = max(right, target_right + guard_x)
    top = min(top, target_top - guard_y)
    bottom = max(bottom, target_bottom + guard_y)

    left = max(0, math.floor(left))
    top = max(0, math.floor(top))
    right = min(width, math.ceil(right))
    bottom = min(height, math.ceil(bottom))
    if right - left < 2 or bottom - top < 2:
        raise ValueError(f"computed invalid crop {(left, top, right, bottom)}")
    box = CropBox(left, top, right, bottom)
    validate_crop_contains_target(box, label, width, height)
    return box


def validate_crop_contains_target(box: CropBox, label: PoseLabel, width: int, height: int) -> None:
    target_left, target_top, target_right, target_bottom = target_bounds(label, width, height)
    tolerance = 1e-6
    if not (
        box.left <= target_left + tolerance
        and box.top <= target_top + tolerance
        and box.right >= target_right - tolerance
        and box.bottom >= target_bottom - tolerance
    ):
        raise ValueError("crop does not contain the original bbox and visible keypoints")


def transform_label(label: PoseLabel, box: CropBox, width: int, height: int) -> PoseLabel:
    cx, cy, bbox_w, bbox_h = label.bbox
    bbox_left = (cx - bbox_w / 2) * width
    bbox_right = (cx + bbox_w / 2) * width
    bbox_top = (cy - bbox_h / 2) * height
    bbox_bottom = (cy + bbox_h / 2) * height
    new_left = (bbox_left - box.left) / box.width
    new_right = (bbox_right - box.left) / box.width
    new_top = (bbox_top - box.top) / box.height
    new_bottom = (bbox_bottom - box.top) / box.height
    new_bbox = (
        (new_left + new_right) / 2,
        (new_top + new_bottom) / 2,
        new_right - new_left,
        new_bottom - new_top,
    )
    new_keypoints = []
    for x, y, visibility in label.keypoints:
        if visibility <= 0:
            new_keypoints.append((0.0, 0.0, visibility))
        else:
            new_keypoints.append(((x * width - box.left) / box.width, (y * height - box.top) / box.height, visibility))
    transformed = PoseLabel(label.class_id, new_bbox, tuple(new_keypoints))
    validate_transformed_label(transformed)
    return transformed


def validate_transformed_label(label: PoseLabel, tolerance: float = 1e-6) -> None:
    cx, cy, bbox_w, bbox_h = label.bbox
    values = (cx, cy, bbox_w, bbox_h)
    if any(value < -tolerance or value > 1 + tolerance for value in values):
        raise ValueError(f"transformed bbox field outside [0,1]: {values}")
    if cx - bbox_w / 2 < -tolerance or cx + bbox_w / 2 > 1 + tolerance:
        raise ValueError("transformed bbox x extent outside crop")
    if cy - bbox_h / 2 < -tolerance or cy + bbox_h / 2 > 1 + tolerance:
        raise ValueError("transformed bbox y extent outside crop")
    for x, y, visibility in label.keypoints:
        if visibility > 0 and not (-tolerance <= x <= 1 + tolerance and -tolerance <= y <= 1 + tolerance):
            raise ValueError(f"transformed visible keypoint outside crop: {(x, y, visibility)}")


def find_train_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"expected train image/label directories below {dataset_root}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    missing = [path.name for path in images if path.stem not in labels]
    extra = sorted(set(labels) - {path.stem for path in images})
    if missing or extra:
        raise ValueError(f"train image/label mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return [(image, labels[image.stem]) for image in images]


def plan_views(
    dataset_root: Path,
    seed: int = 20260822,
    margin: float = 0.20,
    shift_jitter: float = 0.05,
    scale_jitter: float = 0.10,
) -> list[dict]:
    records = []
    for image_path, label_path in find_train_pairs(dataset_root):
        label = parse_pose_label(label_path)
        with Image.open(image_path) as image:
            width, height = image.size
        box = compute_crop_box(label, width, height, image_path.name, seed, margin, shift_jitter, scale_jitter)
        transformed = transform_label(label, box, width, height)
        records.append(
            {
                "source_image": str(image_path),
                "source_label": str(label_path),
                "source_width": width,
                "source_height": height,
                "crop_box": asdict(box),
                "crop_width": box.width,
                "crop_height": box.height,
                "crop_area_fraction": round((box.width * box.height) / (width * height), 8),
                "output_name": f"roi_{image_path.stem}.png",
                "label": transformed,
            }
        )
    return records


def summary_for(records: Sequence[dict]) -> dict:
    area = sorted(record["crop_area_fraction"] for record in records)

    def percentile(fraction: float) -> float | None:
        if not area:
            return None
        return area[round((len(area) - 1) * fraction)]

    return {
        "source_train_count": len(records),
        "roi_view_count": len(records),
        "mixed_train_count": len(records) * 2,
        "crop_area_fraction_median": percentile(0.5),
        "crop_area_fraction_p10": percentile(0.1),
        "crop_area_fraction_p90": percentile(0.9),
        "full_frame_crop_count": sum(record["crop_area_fraction"] >= 0.999 for record in records),
    }


def apply_plan(
    records: Sequence[dict],
    dataset_root: Path,
    output_root: Path,
    seed: int,
    margin: float,
    shift_jitter: float,
    scale_jitter: float,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    image_dir = staging / "images" / "train"
    label_dir = staging / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    manifest_records = []
    try:
        for index, record in enumerate(records, 1):
            source_image = Path(record["source_image"])
            source_label = Path(record["source_label"])
            box = CropBox(**record["crop_box"])
            output_image = image_dir / record["output_name"]
            output_label = label_dir / f"{Path(record['output_name']).stem}.txt"
            with Image.open(source_image) as image:
                image.load()
                cropped = image.crop((box.left, box.top, box.right, box.bottom))
                cropped.save(output_image, format="PNG", compress_level=4)
            label_text = format_pose_label(record["label"])
            output_label.write_text(label_text, encoding="utf-8")
            with Image.open(output_image) as check:
                check.load()
                if check.size != (record["crop_width"], record["crop_height"]):
                    raise ValueError(f"saved crop size mismatch: {output_image}")
            parsed = parse_pose_label(output_label)
            validate_transformed_label(parsed)
            manifest_records.append(
                {
                    "index": index,
                    "source_image": str(source_image.relative_to(dataset_root)),
                    "source_label": str(source_label.relative_to(dataset_root)),
                    "source_width": record["source_width"],
                    "source_height": record["source_height"],
                    "source_image_sha256": sha256_file(source_image),
                    "crop_box": record["crop_box"],
                    "crop_width": record["crop_width"],
                    "crop_height": record["crop_height"],
                    "crop_area_fraction": record["crop_area_fraction"],
                    "output_image": str(output_image.relative_to(staging)),
                    "output_label": str(output_label.relative_to(staging)),
                    "output_image_sha256": sha256_file(output_image),
                    "output_label_sha256": sha256_file(output_label),
                }
            )
        summary = summary_for(records)
        manifest = {
            "schema_version": 1,
            "source_dataset": str(dataset_root),
            "configuration": {
                "seed": seed,
                "margin": margin,
                "shift_jitter": shift_jitter,
                "scale_jitter": scale_jitter,
                "roi_prefix": "roi_",
            },
            "summary": summary,
            "records": manifest_records,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        readme = f"""# 六点Pose ROI裁剪视图

此目录只包含从 `{dataset_root}/images/train` 派生的ROI裁剪训练视图，不包含或替换原图，也不包含val/test。

- 原train：{summary['source_train_count']}张
- ROI视图：{summary['roi_view_count']}张
- 混合YAML训练规模：{summary['mixed_train_count']}张
- ROI面积占原图中位数：{summary['crop_area_fraction_median']:.2%}
- 配置：seed={seed}, margin={margin}, shift_jitter={shift_jitter}, scale_jitter={scale_jitter}

训练使用 `6-train_ap_model/pose_data_roi_mixed.yaml`，其train同时引用原图目录和本目录；val/test仍只引用原始数据。
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        staging.rename(output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/pose_data"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/pose_roi_views"))
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--shift-jitter", type=float, default=0.05)
    parser.add_argument("--scale-jitter", type=float, default=0.10)
    parser.add_argument("--apply", action="store_true", help="write the derived ROI dataset; default is dry-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = plan_views(args.dataset_root, args.seed, args.margin, args.shift_jitter, args.scale_jitter)
    summary = summary_for(records)
    if args.apply:
        manifest = apply_plan(records, args.dataset_root, args.output_root, args.seed, args.margin, args.shift_jitter, args.scale_jitter)
        summary = manifest["summary"]
    else:
        summary = {**summary, "mode": "dry-run", "output_written": False}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
