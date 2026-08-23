#!/usr/bin/env python3
"""Build ROI-only YOLO Pose data from a frozen first-stage model's predicted boxes."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from build_pose_roi_views import (
    CropBox,
    PoseLabel,
    format_pose_label,
    parse_pose_label,
    sha256_file,
    target_bounds,
    transform_label,
    validate_transformed_label,
)
from two_stage_pose_inference import PosePrediction, run_pose_model


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_MODEL = REPO_ROOT / "6-train_ap_model/runs/pose/best_performance-5/weights/best.pt"


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def find_pairs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"missing {split} images/labels below {dataset_root}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    image_stems = {path.stem for path in images}
    missing = [path.name for path in images if path.stem not in labels]
    extra = sorted(set(labels) - image_stems)
    if missing or extra:
        raise ValueError(f"{split} image/label mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return [(image, labels[image.stem]) for image in images]


def deterministic_rng(filename: str, split: str, variant: int, seed: int) -> random.Random:
    import hashlib

    digest = hashlib.sha256(f"{seed}:{split}:{variant}:{filename}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def predicted_crop_box(
    predicted_box: Sequence[float],
    label: PoseLabel,
    width: int,
    height: int,
    filename: str,
    split: str,
    variant: int,
    seed: int = 20260823,
    margin: float = 0.20,
    shift_jitter: float = 0.06,
    scale_jitter: float = 0.12,
    safety_margin: float = 0.03,
    minimum_size: int = 64,
) -> tuple[CropBox, bool]:
    """Create a predicted-box ROI and expand only when needed to keep labels representable."""
    if width < 2 or height < 2:
        raise ValueError(f"invalid image size {width}x{height}")
    x1, y1, x2, y2 = (float(value) for value in predicted_box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid predicted box: {predicted_box}")
    box_w, box_h = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    scale = 1.0
    if variant > 0:
        rng = deterministic_rng(filename, split, variant, seed)
        scale += rng.uniform(-scale_jitter, scale_jitter)
        center_x += rng.uniform(-shift_jitter, shift_jitter) * box_w
        center_y += rng.uniform(-shift_jitter, shift_jitter) * box_h
    crop_w = max(float(minimum_size), box_w * (1.0 + 2.0 * margin) * scale)
    crop_h = max(float(minimum_size), box_h * (1.0 + 2.0 * margin) * scale)
    left = max(0, math.floor(center_x - crop_w / 2))
    top = max(0, math.floor(center_y - crop_h / 2))
    right = min(width, math.ceil(center_x + crop_w / 2))
    bottom = min(height, math.ceil(center_y + crop_h / 2))

    target_left, target_top, target_right, target_bottom = target_bounds(label, width, height)
    target_w = max(1.0, target_right - target_left)
    target_h = max(1.0, target_bottom - target_top)
    guard_x, guard_y = safety_margin * target_w, safety_margin * target_h
    expanded_for_truth = not (
        left <= target_left - guard_x
        and top <= target_top - guard_y
        and right >= target_right + guard_x
        and bottom >= target_bottom + guard_y
    )
    if expanded_for_truth:
        left = max(0, min(left, math.floor(target_left - guard_x)))
        top = max(0, min(top, math.floor(target_top - guard_y)))
        right = min(width, max(right, math.ceil(target_right + guard_x)))
        bottom = min(height, max(bottom, math.ceil(target_bottom + guard_y)))
    if right - left < 2 or bottom - top < 2:
        raise ValueError(f"computed invalid crop: {(left, top, right, bottom)}")
    return CropBox(left, top, right, bottom), expanded_for_truth


def plan_split(
    dataset_root: Path,
    split: str,
    predictor: Callable[[np.ndarray], PosePrediction],
    variants: int,
    seed: int,
    margin: float,
    shift_jitter: float,
    scale_jitter: float,
    minimum_box_confidence: float,
    limit: int | None = None,
) -> tuple[list[dict], list[dict]]:
    if variants < 1:
        raise ValueError("variants must be at least 1")
    pairs = find_pairs(dataset_root, split)
    if limit is not None:
        pairs = pairs[:limit]
    records: list[dict] = []
    skipped: list[dict] = []
    for index, (image_path, label_path) in enumerate(pairs, 1):
        image = load_bgr(image_path)
        height, width = image.shape[:2]
        label = parse_pose_label(label_path)
        prediction = predictor(image)
        if prediction.box_xyxy is None or not prediction.has_keypoints:
            skipped.append({"split": split, "source_image": str(image_path), "reason": "prediction_missing"})
            continue
        if prediction.box_confidence < minimum_box_confidence:
            skipped.append(
                {
                    "split": split,
                    "source_image": str(image_path),
                    "reason": "box_low_confidence",
                    "box_confidence": prediction.box_confidence,
                }
            )
            continue
        for variant in range(variants):
            box, expanded = predicted_crop_box(
                prediction.box_xyxy,
                label,
                width,
                height,
                image_path.name,
                split,
                variant,
                seed,
                margin,
                shift_jitter if split == "train" else 0.0,
                scale_jitter if split == "train" else 0.0,
            )
            transformed = transform_label(label, box, width, height)
            output_stem = f"stage2_{split}_v{variant:02d}_{image_path.stem}"
            records.append(
                {
                    "split": split,
                    "source_index": index,
                    "variant": variant,
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "source_width": width,
                    "source_height": height,
                    "predicted_box_xyxy": tuple(float(value) for value in prediction.box_xyxy),
                    "predicted_box_confidence": float(prediction.box_confidence),
                    "crop_box": asdict(box),
                    "crop_area_fraction": (box.width * box.height) / (width * height),
                    "expanded_for_truth": expanded,
                    "output_name": f"{output_stem}.png",
                    "label": transformed,
                }
            )
        print(f"[{split} {index}/{len(pairs)}] {image_path.name}: {variants} ROI", flush=True)
    return records, skipped


def summarize(records: Sequence[dict], skipped: Sequence[dict]) -> dict:
    areas = sorted(record["crop_area_fraction"] for record in records)

    def percentile(fraction: float) -> float | None:
        return None if not areas else round(areas[round((len(areas) - 1) * fraction)], 8)

    return {
        "source_predictions_used": len({(record["split"], record["source_image"]) for record in records}),
        "roi_view_count": len(records),
        "train_roi_count": sum(record["split"] == "train" for record in records),
        "val_roi_count": sum(record["split"] == "val" for record in records),
        "skipped_source_count": len(skipped),
        "truth_expanded_count": sum(record["expanded_for_truth"] for record in records),
        "truth_expanded_fraction": round(sum(record["expanded_for_truth"] for record in records) / len(records), 8) if records else None,
        "crop_area_fraction_p10": percentile(0.10),
        "crop_area_fraction_median": percentile(0.50),
        "crop_area_fraction_p90": percentile(0.90),
    }


def apply_dataset(
    records: Sequence[dict],
    skipped: Sequence[dict],
    dataset_root: Path,
    output_root: Path,
    stage1_model: Path,
    configuration: dict,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    try:
        manifest_records = []
        for index, record in enumerate(records, 1):
            split = record["split"]
            image_dir = staging / "images" / split
            label_dir = staging / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            source_image = Path(record["source_image"])
            source_label = Path(record["source_label"])
            box = CropBox(**record["crop_box"])
            output_image = image_dir / record["output_name"]
            output_label = label_dir / f"{Path(record['output_name']).stem}.txt"
            with Image.open(source_image) as image:
                image.load()
                image.crop((box.left, box.top, box.right, box.bottom)).save(output_image, format="PNG", compress_level=4)
            output_label.write_text(format_pose_label(record["label"]), encoding="utf-8")
            with Image.open(output_image) as check:
                check.load()
                if check.size != (box.width, box.height):
                    raise ValueError(f"saved crop size mismatch: {output_image}")
            validate_transformed_label(parse_pose_label(output_label))
            manifest_records.append(
                {
                    "index": index,
                    "split": split,
                    "variant": record["variant"],
                    "source_image": str(source_image.relative_to(dataset_root)),
                    "source_label": str(source_label.relative_to(dataset_root)),
                    "source_image_sha256": sha256_file(source_image),
                    "source_label_sha256": sha256_file(source_label),
                    "source_width": record["source_width"],
                    "source_height": record["source_height"],
                    "predicted_box_xyxy": record["predicted_box_xyxy"],
                    "predicted_box_confidence": record["predicted_box_confidence"],
                    "crop_box": record["crop_box"],
                    "crop_area_fraction": round(record["crop_area_fraction"], 8),
                    "expanded_for_truth": record["expanded_for_truth"],
                    "output_image": str(output_image.relative_to(staging)),
                    "output_label": str(output_label.relative_to(staging)),
                    "output_image_sha256": sha256_file(output_image),
                    "output_label_sha256": sha256_file(output_label),
                }
            )
        summary = summarize(records, skipped)
        manifest = {
            "schema_version": 1,
            "purpose": "stage2_pose_finetuning_from_first_stage_predicted_rois",
            "source_dataset": str(dataset_root.resolve()),
            "stage1_model": str(stage1_model.resolve()),
            "stage1_model_sha256": sha256_file(stage1_model),
            "configuration": configuration,
            "summary": summary,
            "skipped": skipped,
            "records": manifest_records,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(
            "# 六点Pose二阶段预测ROI数据集\n\n"
            "此派生集由冻结的一阶段模型预测框生成，只用于二阶段Pose精修。"
            "train包含确定性多ROI视图，val只包含线上同margin的单ROI；不包含test。\n\n"
            f"- train ROI：{summary['train_roi_count']}\n"
            f"- val ROI：{summary['val_roi_count']}\n"
            f"- 跳过源图：{summary['skipped_source_count']}\n"
            f"- 为保持GT可表示而扩框：{summary['truth_expanded_count']} ({summary['truth_expanded_fraction']:.2%})\n"
            f"- 一阶段模型SHA-256：`{manifest['stage1_model_sha256']}`\n\n"
            "训练入口：`6-train_ap_model/train_pose_stage2.py`。最终模型选择必须使用原始val完整两阶段链路。\n",
            encoding="utf-8",
        )
        staging.rename(output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets/pose_data")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "datasets/pose_stage2_roi")
    parser.add_argument("--stage1-model", type=Path, default=DEFAULT_STAGE1_MODEL)
    parser.add_argument("--train-variants", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--shift-jitter", type=float, default=0.06)
    parser.add_argument("--scale-jitter", type=float, default=0.12)
    parser.add_argument("--minimum-box-confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--limit", type=int, help="smoke-test only; process at most N sources per split")
    parser.add_argument("--apply", action="store_true", help="write the derived dataset; default only plans and summarizes")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.apply and args.limit is not None:
        raise ValueError("--limit cannot be combined with --apply")
    if not args.stage1_model.is_file():
        raise FileNotFoundError(f"first-stage model does not exist: {args.stage1_model}")
    if args.apply and args.output_root.exists():
        raise FileExistsError(f"output already exists: {args.output_root}")
    from ultralytics import YOLO

    model = YOLO(str(args.stage1_model))

    def predictor(image: np.ndarray) -> PosePrediction:
        return run_pose_model(model, image, args.minimum_box_confidence, args.imgsz, args.device)

    train, train_skipped = plan_split(
        args.dataset_root,
        "train",
        predictor,
        args.train_variants,
        args.seed,
        args.margin,
        args.shift_jitter,
        args.scale_jitter,
        args.minimum_box_confidence,
        args.limit,
    )
    val, val_skipped = plan_split(
        args.dataset_root,
        "val",
        predictor,
        1,
        args.seed,
        args.margin,
        0.0,
        0.0,
        args.minimum_box_confidence,
        args.limit,
    )
    records = [*train, *val]
    skipped = [*train_skipped, *val_skipped]
    configuration = {
        "seed": args.seed,
        "train_variants": args.train_variants,
        "production_margin": args.margin,
        "train_shift_jitter": args.shift_jitter,
        "train_scale_jitter": args.scale_jitter,
        "minimum_box_confidence": args.minimum_box_confidence,
        "imgsz": args.imgsz,
        "device": args.device,
    }
    summary = summarize(records, skipped)
    if args.apply:
        summary = apply_dataset(records, skipped, args.dataset_root, args.output_root, args.stage1_model, configuration)["summary"]
    else:
        summary = {**summary, "mode": "dry-run", "output_written": False}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
