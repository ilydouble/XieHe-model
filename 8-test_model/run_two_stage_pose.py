#!/usr/bin/env python3
"""Run local single-vs-two-stage six-point Pose inference and optional label metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from two_stage_pose_inference import (  # noqa: E402
    KEYPOINT_COUNT,
    PosePrediction,
    prediction_to_dict,
    run_pose_model,
    result_to_dict,
    two_stage_predict,
)


KEYPOINT_NAMES = ("CR", "CL", "IR", "IL", "SR", "SL")
COLORS = ((255, 150, 30), (255, 210, 30), (40, 210, 80), (30, 210, 180), (60, 70, 255), (80, 150, 255))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def parse_label(path: Path, width: int, height: int) -> tuple[tuple[float, float, int], ...]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0].split()) != 23:
        raise ValueError(f"invalid six-point label: {path}")
    values = [float(value) for value in lines[0].split()[5:]]
    return tuple((values[i] * width, values[i + 1] * height, int(values[i + 2])) for i in range(0, 18, 3))


def prediction_metrics(prediction: PosePrediction, truth: tuple[tuple[float, float, int], ...]) -> dict:
    per_point = []
    for name, predicted, confidence, target in zip(KEYPOINT_NAMES, prediction.keypoints_xy, prediction.keypoint_confidences, truth):
        tx, ty, visibility = target
        px, py = predicted
        measured = visibility > 0 and confidence > 0
        per_point.append(
            {
                "name": name,
                "measured": measured,
                "dx_px": None if not measured else px - tx,
                "dy_px": None if not measured else py - ty,
                "error_px": None if not measured else float(np.hypot(px - tx, py - ty)),
            }
        )
    measured = [point for point in per_point if point["measured"]]
    shoulders = [point for point in measured if point["name"] in ("CR", "CL")]
    lower = [point for point in measured if point["name"] not in ("CR", "CL")]

    def mean(field: str, values: list[dict]) -> float | None:
        return None if not values else float(sum(point[field] for point in values) / len(values))

    return {
        "detected": len(measured),
        "mean_error_px": mean("error_px", measured),
        "mean_dy_px": mean("dy_px", measured),
        "shoulder_mean_dy_px": mean("dy_px", shoulders),
        "lower_mean_dy_px": mean("dy_px", lower),
        "per_point": per_point,
    }


def draw_prediction(image: np.ndarray, prediction: PosePrediction, title: str, roi=None) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    line = max(2, round(min(width, height) / 500))
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 210, 255), line)
    if prediction.box_xyxy is not None:
        x1, y1, x2, y2 = (round(value) for value in prediction.box_xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), (90, 255, 90), line)
    radius = max(5, round(min(width, height) / 180))
    for name, color, (x, y), confidence in zip(KEYPOINT_NAMES, COLORS, prediction.keypoints_xy, prediction.keypoint_confidences):
        if confidence <= 0:
            continue
        point = round(x), round(y)
        cv2.circle(output, point, radius + 2, (255, 255, 255), -1)
        cv2.circle(output, point, radius, color, -1)
        cv2.putText(output, name, (point[0] + radius + 3, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.rectangle(output, (0, 0), (width, 48), (25, 25, 25), -1)
    cv2.putText(output, title, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return output


def fit_height(image: np.ndarray, target_height: int = 1000) -> np.ndarray:
    if image.shape[0] <= target_height:
        return image
    scale = target_height / image.shape[0]
    return cv2.resize(image, (round(image.shape[1] * scale), target_height), interpolation=cv2.INTER_AREA)


def make_comparison(image: np.ndarray, first: PosePrediction, final: PosePrediction, roi, used: bool, reason: str | None) -> np.ndarray:
    left = fit_height(draw_prediction(image, first, "Stage 1: full image", roi))
    status = "Stage 2: ROI refined" if used else f"Fallback: {reason}"
    right = fit_height(draw_prediction(image, final, status, roi))
    target_h = max(left.shape[0], right.shape[0])
    if left.shape[0] != target_h:
        left = cv2.resize(left, (round(left.shape[1] * target_h / left.shape[0]), target_h))
    if right.shape[0] != target_h:
        right = cv2.resize(right, (round(right.shape[1] * target_h / right.shape[0]), target_h))
    separator = np.full((target_h, 8, 3), 180, dtype=np.uint8)
    return np.hstack((left, separator, right))


def find_images(image: Path | None, image_dir: Path | None, keyword: str) -> list[Path]:
    if image:
        return [image]
    return [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES and (not keyword or keyword in path.name)]


def aggregate(samples: list[dict], key: str) -> dict | None:
    metrics = [sample[key] for sample in samples if sample.get(key)]
    if not metrics:
        return None
    fields = ("mean_error_px", "mean_dy_px", "shoulder_mean_dy_px", "lower_mean_dy_px")
    return {
        "sample_count": len(metrics),
        "all_six_detected": sum(metric["detected"] == KEYPOINT_COUNT for metric in metrics),
        **{
            field: sum(metric[field] for metric in metrics if metric[field] is not None) / sum(metric[field] is not None for metric in metrics)
            for field in fields
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--image-dir", type=Path)
    parser.add_argument("--label-dir", type=Path, help="optional YOLO labels for quantitative comparison")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keyword", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--roi-margin", type=float, default=0.20)
    parser.add_argument("--roi-conf", type=float, default=0.25)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir()
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    samples = []
    for index, image_path in enumerate(find_images(args.image, args.image_dir, args.keyword), 1):
        image = load_bgr(image_path)
        result = two_stage_predict(
            model,
            image,
            confidence=args.conf,
            image_size=args.imgsz,
            roi_margin=args.roi_margin,
            minimum_first_box_confidence=args.roi_conf,
            device=args.device,
        )
        preview = preview_dir / f"{index:04d}_{image_path.stem}.jpg"
        cv2.imwrite(str(preview), make_comparison(image, result.first, result.final, result.roi_xyxy, result.used_second_stage, result.fallback_reason), [cv2.IMWRITE_JPEG_QUALITY, 90])
        sample = {"filename": image_path.name, **result_to_dict(result), "preview": str(preview.relative_to(args.output_dir))}
        if args.label_dir:
            label_path = args.label_dir / f"{image_path.stem}.txt"
            truth = parse_label(label_path, image.shape[1], image.shape[0])
            sample["first_metrics"] = prediction_metrics(result.first, truth)
            sample["final_metrics"] = prediction_metrics(result.final, truth)
        samples.append(sample)
        print(f"[{index}] {image_path.name}: second_stage={result.used_second_stage} fallback={result.fallback_reason}", flush=True)
    summary = {
        "sample_count": len(samples),
        "second_stage_used": sum(sample["used_second_stage"] for sample in samples),
        "fallbacks": {},
        "first_metrics": aggregate(samples, "first_metrics"),
        "final_metrics": aggregate(samples, "final_metrics"),
    }
    for sample in samples:
        if sample["fallback_reason"]:
            summary["fallbacks"][sample["fallback_reason"]] = summary["fallbacks"].get(sample["fallback_reason"], 0) + 1
    payload = {"model": str(args.model.resolve()), "configuration": vars(args), "summary": summary, "samples": samples}
    payload["configuration"] = {key: str(value) if isinstance(value, Path) else value for key, value in payload["configuration"].items()}
    (args.output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "used_second_stage", "fallback_reason", "first_mean_error_px", "final_mean_error_px", "first_shoulder_dy_px", "final_shoulder_dy_px", "first_lower_dy_px", "final_lower_dy_px", "preview"))
        for sample in samples:
            first, final = sample.get("first_metrics") or {}, sample.get("final_metrics") or {}
            writer.writerow((sample["filename"], sample["used_second_stage"], sample["fallback_reason"] or "", first.get("mean_error_px"), final.get("mean_error_px"), first.get("shoulder_mean_dy_px"), final.get("shoulder_mean_dy_px"), first.get("lower_mean_dy_px"), final.get("lower_mean_dy_px"), sample["preview"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
