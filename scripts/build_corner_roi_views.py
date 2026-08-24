#!/usr/bin/env python3
"""Build an incremental mixed-training ROI layer for the vertebra corner model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BASE_CORNER_CLASS_IDS = frozenset(range(18))
OPTIONAL_CORNER_CLASS_IDS = frozenset({18, 19})
ALLOWED_CORNER_CLASS_IDS = BASE_CORNER_CLASS_IDS | OPTIONAL_CORNER_CLASS_IDS


@dataclass(frozen=True)
class CornerObject:
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


def parse_corner_label(path: Path) -> tuple[CornerObject, ...]:
    objects = []
    seen_classes = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 17:
            raise ValueError(f"{path}:{line_number}: expected 17 fields, found {len(fields)}")
        class_value = float(fields[0])
        if not class_value.is_integer() or int(class_value) not in ALLOWED_CORNER_CLASS_IDS:
            raise ValueError(f"{path}:{line_number}: invalid class id {fields[0]}")
        class_id = int(class_value)
        if class_id in seen_classes:
            raise ValueError(f"{path}:{line_number}: duplicate class id {class_id}")
        seen_classes.add(class_id)
        bbox = tuple(float(value) for value in fields[1:5])
        if any(not 0.0 <= value <= 1.0 for value in bbox):
            raise ValueError(f"{path}:{line_number}: bbox field outside [0,1]")
        keypoints = []
        for offset in range(5, 17, 3):
            x, y, visibility = (float(value) for value in fields[offset : offset + 3])
            if visibility not in (0.0, 1.0, 2.0):
                raise ValueError(f"{path}:{line_number}: invalid visibility {visibility}")
            if visibility > 0 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"{path}:{line_number}: visible keypoint outside [0,1]")
            keypoints.append((x, y, int(visibility)))
        objects.append(CornerObject(class_id, bbox, tuple(keypoints)))
    if not objects:
        raise ValueError(f"{path}: no corner objects")
    return tuple(objects)


def format_corner_label(objects: Sequence[CornerObject]) -> str:
    lines = []
    for item in objects:
        fields = [str(item.class_id), *(f"{value:.8f}" for value in item.bbox)]
        for x, y, visibility in item.keypoints:
            fields.extend((f"{x:.8f}", f"{y:.8f}", str(visibility)))
        lines.append(" ".join(fields))
    return "\n".join(lines) + "\n"


def target_bounds(objects: Sequence[CornerObject], width: int, height: int) -> tuple[float, float, float, float]:
    left, top, right, bottom = float(width), float(height), 0.0, 0.0
    visible_count = 0
    for item in objects:
        cx, cy, bbox_w, bbox_h = item.bbox
        left = min(left, (cx - bbox_w / 2) * width)
        right = max(right, (cx + bbox_w / 2) * width)
        top = min(top, (cy - bbox_h / 2) * height)
        bottom = max(bottom, (cy + bbox_h / 2) * height)
        for x, y, visibility in item.keypoints:
            if visibility <= 0:
                continue
            visible_count += 1
            left, right = min(left, x * width), max(right, x * width)
            top, bottom = min(top, y * height), max(bottom, y * height)
    if visible_count == 0:
        raise ValueError("corner label has no visible keypoints")
    return max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom)


def crop_contains_target(box: CropBox, objects: Sequence[CornerObject], width: int, height: int) -> bool:
    left, top, right, bottom = target_bounds(objects, width, height)
    tolerance = 1e-6
    return (
        box.left <= left + tolerance
        and box.top <= top + tolerance
        and box.right >= right - tolerance
        and box.bottom >= bottom - tolerance
    )


def deterministic_rng(filename: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{filename}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def synthetic_crop_box(
    objects: Sequence[CornerObject],
    width: int,
    height: int,
    filename: str,
    seed: int,
    horizontal_margin: float,
    vertical_margin: float,
    shift_jitter: float,
    scale_jitter: float,
    safety_margin: float,
) -> CropBox:
    target_left, target_top, target_right, target_bottom = target_bounds(objects, width, height)
    target_w = max(1.0, target_right - target_left)
    target_h = max(1.0, target_bottom - target_top)
    rng = deterministic_rng(filename, seed)
    scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    crop_w = max(64.0, target_w * (1.0 + 2.0 * horizontal_margin) * scale)
    crop_h = max(64.0, target_h * (1.0 + 2.0 * vertical_margin) * scale)
    center_x = (target_left + target_right) / 2 + rng.uniform(-shift_jitter, shift_jitter) * target_w
    center_y = (target_top + target_bottom) / 2 + rng.uniform(-shift_jitter, shift_jitter) * target_h
    left, right = center_x - crop_w / 2, center_x + crop_w / 2
    top, bottom = center_y - crop_h / 2, center_y + crop_h / 2
    guard_x, guard_y = safety_margin * target_w, safety_margin * target_h
    left, right = min(left, target_left - guard_x), max(right, target_right + guard_x)
    top, bottom = min(top, target_top - guard_y), max(bottom, target_bottom + guard_y)
    box = CropBox(max(0, math.floor(left)), max(0, math.floor(top)), min(width, math.ceil(right)), min(height, math.ceil(bottom)))
    validate_crop(box, objects, width, height)
    return box


def expand_existing_crop(
    existing: CropBox,
    objects: Sequence[CornerObject],
    width: int,
    height: int,
    safety_margin: float,
) -> CropBox:
    target_left, target_top, target_right, target_bottom = target_bounds(objects, width, height)
    target_w = max(1.0, target_right - target_left)
    target_h = max(1.0, target_bottom - target_top)
    guard_x, guard_y = safety_margin * target_w, safety_margin * target_h
    box = CropBox(
        max(0, math.floor(min(existing.left, target_left - guard_x))),
        max(0, math.floor(min(existing.top, target_top - guard_y))),
        min(width, math.ceil(max(existing.right, target_right + guard_x))),
        min(height, math.ceil(max(existing.bottom, target_bottom + guard_y))),
    )
    validate_crop(box, objects, width, height)
    return box


def validate_crop(box: CropBox, objects: Sequence[CornerObject], width: int, height: int) -> None:
    if not (0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height):
        raise ValueError(f"invalid crop {box} for {width}x{height}")
    if box.width < 2 or box.height < 2:
        raise ValueError(f"crop is too small: {box}")
    if not crop_contains_target(box, objects, width, height):
        raise ValueError("crop does not contain every corner bbox and visible keypoint")


def transform_objects(objects: Sequence[CornerObject], box: CropBox, width: int, height: int) -> tuple[CornerObject, ...]:
    transformed = []
    for item in objects:
        cx, cy, bbox_w, bbox_h = item.bbox
        bbox_left = (cx - bbox_w / 2) * width
        bbox_right = (cx + bbox_w / 2) * width
        bbox_top = (cy - bbox_h / 2) * height
        bbox_bottom = (cy + bbox_h / 2) * height
        new_left, new_right = (bbox_left - box.left) / box.width, (bbox_right - box.left) / box.width
        new_top, new_bottom = (bbox_top - box.top) / box.height, (bbox_bottom - box.top) / box.height
        new_bbox = ((new_left + new_right) / 2, (new_top + new_bottom) / 2, new_right - new_left, new_bottom - new_top)
        keypoints = []
        for x, y, visibility in item.keypoints:
            if visibility <= 0:
                keypoints.append((0.0, 0.0, visibility))
            else:
                keypoints.append(((x * width - box.left) / box.width, (y * height - box.top) / box.height, visibility))
        transformed.append(CornerObject(item.class_id, new_bbox, tuple(keypoints)))
    result = tuple(transformed)
    validate_transformed_objects(result)
    return result


def validate_transformed_objects(objects: Sequence[CornerObject], tolerance: float = 2e-6) -> None:
    for item in objects:
        cx, cy, bbox_w, bbox_h = item.bbox
        if any(value < -tolerance or value > 1 + tolerance for value in item.bbox):
            raise ValueError(f"class {item.class_id}: transformed bbox field outside [0,1]")
        if cx - bbox_w / 2 < -tolerance or cx + bbox_w / 2 > 1 + tolerance:
            raise ValueError(f"class {item.class_id}: transformed bbox x extent outside crop")
        if cy - bbox_h / 2 < -tolerance or cy + bbox_h / 2 > 1 + tolerance:
            raise ValueError(f"class {item.class_id}: transformed bbox y extent outside crop")
        for x, y, visibility in item.keypoints:
            if visibility > 0 and not (-tolerance <= x <= 1 + tolerance and -tolerance <= y <= 1 + tolerance):
                raise ValueError(f"class {item.class_id}: transformed keypoint outside crop")


def find_train_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    missing = [path.name for path in images if path.stem not in labels]
    extra = sorted(set(labels) - {path.stem for path in images})
    if missing or extra:
        raise ValueError(f"train image/label mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return [(image, labels[image.stem]) for image in images]


def load_pose_roi_records(pose_roi_root: Path) -> dict[str, dict]:
    manifest_path = pose_roi_root / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    for record in data.get("records", []):
        stem = Path(record["source_image"]).stem
        if stem in records:
            raise ValueError(f"duplicate Pose ROI source stem: {stem}")
        records[stem] = record
    return records


def record_signature(record: dict, configuration: dict) -> str:
    payload = {
        "source_image_sha256": record["source_image_sha256"],
        "source_label_sha256": record["source_label_sha256"],
        "crop_box": record["crop_box"],
        "output_mode": record["output_mode"],
        "reuse_image_sha256": record.get("reuse_image_sha256"),
        "configuration": configuration,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def plan_views(
    corner_root: Path,
    pose_roi_root: Path,
    seed: int = 20260822,
    horizontal_margin: float = 1.40,
    vertical_margin: float = 0.15,
    shift_jitter: float = 0.05,
    scale_jitter: float = 0.10,
    safety_margin: float = 0.03,
) -> tuple[list[dict], dict]:
    configuration = {
        "seed": seed,
        "horizontal_margin": horizontal_margin,
        "vertical_margin": vertical_margin,
        "shift_jitter": shift_jitter,
        "scale_jitter": scale_jitter,
        "safety_margin": safety_margin,
        "roi_prefix": "roi_",
    }
    pose_records = load_pose_roi_records(pose_roi_root)
    records = []
    verified_pose_images = {}
    for image_path, label_path in find_train_pairs(corner_root):
        objects = parse_corner_label(label_path)
        with Image.open(image_path) as image:
            width, height = image.size
        source_image_sha = sha256_file(image_path)
        source_label_sha = sha256_file(label_path)
        pose_record = pose_records.get(image_path.stem)
        reuse_path = None
        reuse_sha = None
        if pose_record and (
            pose_record.get("source_width") == width
            and pose_record.get("source_height") == height
            and pose_record.get("source_image_sha256") == source_image_sha
        ):
            existing = CropBox(**pose_record["crop_box"])
            reuse_path = pose_roi_root / pose_record["output_image"]
            if not reuse_path.is_file():
                raise FileNotFoundError(f"Pose ROI image is missing: {reuse_path}")
            expected_reuse_sha = pose_record["output_image_sha256"]
            if expected_reuse_sha not in verified_pose_images:
                actual_reuse_sha = sha256_file(reuse_path)
                if actual_reuse_sha != expected_reuse_sha:
                    raise ValueError(f"Pose ROI image hash mismatch: {reuse_path}")
                verified_pose_images[expected_reuse_sha] = actual_reuse_sha
            reuse_sha = expected_reuse_sha
            if crop_contains_target(existing, objects, width, height):
                box = existing
                output_mode = "hardlink"
                plan_reason = "reused_pose_roi"
            else:
                box = expand_existing_crop(existing, objects, width, height, safety_margin)
                output_mode = "generated"
                plan_reason = "expanded_unsafe_pose_roi"
                reuse_path = None
                reuse_sha = None
        else:
            box = synthetic_crop_box(
                objects,
                width,
                height,
                image_path.name,
                seed,
                horizontal_margin,
                vertical_margin,
                shift_jitter,
                scale_jitter,
                safety_margin,
            )
            output_mode = "generated"
            plan_reason = "missing_pose_roi" if not pose_record else "pose_source_mismatch"
        transformed = transform_objects(objects, box, width, height)
        record = {
            "source_image": str(image_path.relative_to(corner_root)),
            "source_label": str(label_path.relative_to(corner_root)),
            "source_width": width,
            "source_height": height,
            "source_image_sha256": source_image_sha,
            "source_label_sha256": source_label_sha,
            "object_count": len(objects),
            "visible_keypoint_count": sum(v > 0 for item in objects for _, _, v in item.keypoints),
            "crop_box": asdict(box),
            "crop_width": box.width,
            "crop_height": box.height,
            "crop_area_fraction": round((box.width * box.height) / (width * height), 8),
            "output_image": f"images/train/roi_{image_path.stem}.png",
            "output_label": f"labels/train/roi_{image_path.stem}.txt",
            "output_mode": output_mode,
            "plan_reason": plan_reason,
            "reuse_image": None if reuse_path is None else str(reuse_path.relative_to(pose_roi_root)),
            "reuse_image_sha256": reuse_sha,
            "_source_image_path": image_path,
            "_reuse_image_path": reuse_path,
            "_transformed_objects": transformed,
        }
        record["signature"] = record_signature(record, configuration)
        records.append(record)
    return records, configuration


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summary_for(records: Sequence[dict], skipped_existing: int = 0) -> dict:
    areas = [record["crop_area_fraction"] for record in records]
    return {
        "source_train_count": len(records),
        "roi_view_count": len(records),
        "mixed_train_count": len(records) * 2,
        "reused_pose_roi_hardlink_count": sum(record["plan_reason"] == "reused_pose_roi" for record in records),
        "expanded_unsafe_pose_roi_count": sum(record["plan_reason"] == "expanded_unsafe_pose_roi" for record in records),
        "missing_pose_roi_generated_count": sum(record["plan_reason"] == "missing_pose_roi" for record in records),
        "pose_source_mismatch_generated_count": sum(record["plan_reason"] == "pose_source_mismatch" for record in records),
        "new_pixel_file_count": sum(record["output_mode"] == "generated" for record in records),
        "skipped_existing_count": skipped_existing,
        "crop_area_fraction_median": percentile(areas, 0.5),
        "crop_area_fraction_p10": percentile(areas, 0.1),
        "crop_area_fraction_p90": percentile(areas, 0.9),
        "full_frame_crop_count": sum(area >= 0.999 for area in areas),
    }


def public_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def output_is_current(record: dict, old: dict, output_root: Path) -> bool:
    if old.get("signature") != record["signature"]:
        return False
    output_image = output_root / record["output_image"]
    output_label = output_root / record["output_label"]
    if not output_image.is_file() or not output_label.is_file():
        return False
    if sha256_file(output_label) != old.get("output_label_sha256"):
        return False
    if record["output_mode"] == "hardlink":
        reuse_path = record["_reuse_image_path"]
        return reuse_path is not None and os.path.samefile(output_image, reuse_path)
    return sha256_file(output_image) == old.get("output_image_sha256")


def write_output_record(record: dict, output_root: Path) -> dict:
    output_image = output_root / record["output_image"]
    output_label = output_root / record["output_label"]
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)

    if record["output_mode"] == "hardlink":
        reuse_path = record["_reuse_image_path"]
        fd, temporary_name = tempfile.mkstemp(prefix=f".{output_image.name}.", dir=output_image.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        temporary.unlink()
        try:
            os.link(reuse_path, temporary)
            os.replace(temporary, output_image)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{output_image.stem}.", suffix=".png", dir=output_image.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            box = CropBox(**record["crop_box"])
            with Image.open(record["_source_image_path"]) as image:
                image.load()
                image.crop((box.left, box.top, box.right, box.bottom)).save(temporary, format="PNG", compress_level=4)
            with Image.open(temporary) as check:
                check.load()
                if check.size != (record["crop_width"], record["crop_height"]):
                    raise ValueError(f"saved crop size mismatch: {temporary}")
            os.replace(temporary, output_image)
        finally:
            temporary.unlink(missing_ok=True)

    label_text = format_corner_label(record["_transformed_objects"])
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_label.name}.", dir=output_label.parent, text=True)
    os.close(fd)
    temporary_label = Path(temporary_name)
    try:
        temporary_label.write_text(label_text, encoding="utf-8")
        validate_transformed_objects(parse_corner_label(temporary_label))
        os.replace(temporary_label, output_label)
    finally:
        temporary_label.unlink(missing_ok=True)

    result = public_record(record)
    result["output_image_sha256"] = sha256_file(output_image)
    result["output_label_sha256"] = sha256_file(output_label)
    return result


def assert_no_stale_outputs(records: Sequence[dict], output_root: Path) -> None:
    expected_images = {record["output_image"] for record in records}
    expected_labels = {record["output_label"] for record in records}
    actual_images = {
        str(path.relative_to(output_root))
        for path in (output_root / "images" / "train").glob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    actual_labels = {str(path.relative_to(output_root)) for path in (output_root / "labels" / "train").glob("*.txt")}
    stale = sorted((actual_images - expected_images) | (actual_labels - expected_labels))
    if stale:
        raise ValueError(f"stale output files are not removed automatically: {stale[:10]}")


def write_metadata(output_root: Path, manifest: dict) -> None:
    summary = manifest["summary"]
    readme = f"""# Corner ROI增量训练视图

此目录为`pose_corner_data`的派生训练层，不包含val/test，也不替代原图。

- Corner原train：{summary['source_train_count']}张
- ROI视图：{summary['roi_view_count']}张
- 复用Pose ROI硬链接：{summary['reused_pose_roi_hardlink_count']}张
- 新写图像像素：{summary['new_pixel_file_count']}张
- 混合训练规模：{summary['mixed_train_count']}张
- ROI面积中位数：{summary['crop_area_fraction_median']:.2%}

训练使用`6-train_ap_model/corner_data_roi_mixed.yaml`。硬链接复用图像不重复占用本地数据块；复制或上传时应使用保留硬链接的归档方式。
"""
    for name, content in (("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"), ("README.md", readme)):
        destination = output_root / name
        fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=output_root, text=True)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def apply_plan(records: Sequence[dict], configuration: dict, corner_root: Path, pose_roi_root: Path, output_root: Path) -> dict:
    first_build = not output_root.exists()
    if first_build:
        work_root = output_root.with_name(output_root.name + ".building")
        if work_root.exists():
            raise FileExistsError(f"staging directory already exists: {work_root}")
        (work_root / "images" / "train").mkdir(parents=True)
        (work_root / "labels" / "train").mkdir(parents=True)
        old_records = {}
    else:
        work_root = output_root
        manifest_path = work_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"existing output has no incremental manifest: {output_root}")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_records = {Path(record["source_image"]).stem: record for record in old_manifest.get("records", [])}
        assert_no_stale_outputs(records, output_root)

    manifest_records = []
    skipped = 0
    try:
        for record in records:
            old = old_records.get(Path(record["source_image"]).stem)
            if not first_build and old and output_is_current(record, old, work_root):
                manifest_records.append(old)
                skipped += 1
            else:
                manifest_records.append(write_output_record(record, work_root))
        summary = summary_for(records, skipped)
        manifest = {
            "schema_version": 1,
            "source_dataset": str(corner_root),
            "pose_roi_dataset": str(pose_roi_root),
            "configuration": configuration,
            "summary": summary,
            "records": manifest_records,
        }
        write_metadata(work_root, manifest)
        if first_build:
            work_root.rename(output_root)
        return manifest
    except Exception:
        if first_build:
            shutil.rmtree(work_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corner-root", type=Path, default=Path("datasets/pose_corner_data"))
    parser.add_argument("--pose-roi-root", type=Path, default=Path("datasets/pose_roi_views"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/corner_roi_views"))
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--horizontal-margin", type=float, default=1.40)
    parser.add_argument("--vertical-margin", type=float, default=0.15)
    parser.add_argument("--shift-jitter", type=float, default=0.05)
    parser.add_argument("--scale-jitter", type=float, default=0.10)
    parser.add_argument("--safety-margin", type=float, default=0.03)
    parser.add_argument("--apply", action="store_true", help="write or incrementally update outputs; default is dry-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, configuration = plan_views(
        args.corner_root,
        args.pose_roi_root,
        args.seed,
        args.horizontal_margin,
        args.vertical_margin,
        args.shift_jitter,
        args.scale_jitter,
        args.safety_margin,
    )
    if args.apply:
        summary = apply_plan(records, configuration, args.corner_root, args.pose_roi_root, args.output_root)["summary"]
    else:
        summary = {**summary_for(records), "mode": "dry-run", "output_written": False}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
