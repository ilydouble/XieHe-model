#!/usr/bin/env python3
"""Reusable two-stage YOLO Pose inference with ROI remapping and safe fallback."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


KEYPOINT_COUNT = 6


@dataclass(frozen=True)
class PosePrediction:
    image_width: int
    image_height: int
    box_xyxy: tuple[float, float, float, float] | None
    box_confidence: float
    keypoints_xy: tuple[tuple[float, float], ...]
    keypoint_confidences: tuple[float, ...]

    @property
    def has_keypoints(self) -> bool:
        return any(confidence > 0 for confidence in self.keypoint_confidences)

    def normalized_keypoints(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            (x / self.image_width, y / self.image_height, confidence)
            for (x, y), confidence in zip(self.keypoints_xy, self.keypoint_confidences)
        )


@dataclass(frozen=True)
class TwoStageResult:
    first: PosePrediction
    final: PosePrediction
    used_second_stage: bool
    fallback_reason: str | None
    roi_xyxy: tuple[int, int, int, int] | None
    first_inference_ms: float
    second_inference_ms: float | None
    total_inference_ms: float


def empty_prediction(width: int, height: int) -> PosePrediction:
    return PosePrediction(
        width,
        height,
        None,
        0.0,
        tuple((0.0, 0.0) for _ in range(KEYPOINT_COUNT)),
        tuple(0.0 for _ in range(KEYPOINT_COUNT)),
    )


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_prediction(result: Any, width: int, height: int) -> PosePrediction:
    if result is None or getattr(result, "keypoints", None) is None:
        return empty_prediction(width, height)
    keypoints = to_numpy(getattr(result.keypoints, "xy", None))
    if keypoints.ndim != 3 or keypoints.shape[0] == 0:
        return empty_prediction(width, height)

    boxes = getattr(result, "boxes", None)
    box_confidences = to_numpy(getattr(boxes, "conf", None)).reshape(-1) if boxes is not None else np.asarray([])
    boxes_xyxy = to_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else np.asarray([])
    best = int(np.argmax(box_confidences)) if box_confidences.size else 0
    best = min(best, keypoints.shape[0] - 1)
    box = None
    box_confidence = 0.0
    if boxes_xyxy.ndim == 2 and best < boxes_xyxy.shape[0] and boxes_xyxy.shape[1] >= 4:
        box = tuple(float(value) for value in boxes_xyxy[best, :4])
        box_confidence = float(box_confidences[best]) if best < box_confidences.size else 0.0

    coordinates = keypoints[best]
    confidence_values = to_numpy(getattr(result.keypoints, "conf", None))
    if confidence_values.ndim == 2 and best < confidence_values.shape[0]:
        selected_confidences = confidence_values[best]
    else:
        selected_confidences = np.ones(len(coordinates), dtype=float)
    points = []
    confidences = []
    for index in range(KEYPOINT_COUNT):
        if index < len(coordinates):
            x, y = coordinates[index, :2]
            points.append((float(x), float(y)))
            confidences.append(float(selected_confidences[index]) if index < len(selected_confidences) else 1.0)
        else:
            points.append((0.0, 0.0))
            confidences.append(0.0)
    return PosePrediction(width, height, box, box_confidence, tuple(points), tuple(confidences))


def run_pose_model(model: Any, image: np.ndarray, confidence: float, image_size: int, device: str | None = None) -> PosePrediction:
    height, width = image.shape[:2]
    kwargs = {"conf": confidence, "imgsz": image_size, "verbose": False}
    if device:
        kwargs["device"] = device
    results = model.predict(image, **kwargs)
    return extract_prediction(results[0] if results else None, width, height)


def expand_roi(
    box_xyxy: Sequence[float],
    width: int,
    height: int,
    margin: float = 0.20,
    minimum_side: int = 64,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    box_w, box_h = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    roi_w = max(float(minimum_side), box_w * (1 + 2 * margin))
    roi_h = max(float(minimum_side), box_h * (1 + 2 * margin))
    left = max(0, math.floor(center_x - roi_w / 2))
    top = max(0, math.floor(center_y - roi_h / 2))
    right = min(width, math.ceil(center_x + roi_w / 2))
    bottom = min(height, math.ceil(center_y + roi_h / 2))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def remap_prediction(prediction: PosePrediction, roi_xyxy: Sequence[int], original_width: int, original_height: int) -> PosePrediction:
    left, top, _, _ = roi_xyxy
    box = None
    if prediction.box_xyxy is not None:
        x1, y1, x2, y2 = prediction.box_xyxy
        box = (x1 + left, y1 + top, x2 + left, y2 + top)
    points = tuple((x + left, y + top) for x, y in prediction.keypoints_xy)
    return PosePrediction(
        original_width,
        original_height,
        box,
        prediction.box_confidence,
        points,
        prediction.keypoint_confidences,
    )


def two_stage_predict(
    model: Any,
    image: np.ndarray,
    confidence: float = 0.25,
    image_size: int = 800,
    roi_margin: float = 0.20,
    minimum_first_box_confidence: float = 0.25,
    minimum_roi_side: int = 64,
    maximum_roi_area_fraction: float = 0.98,
    device: str | None = None,
    second_model: Any | None = None,
) -> TwoStageResult:
    total_start = time.perf_counter()
    height, width = image.shape[:2]
    first_start = time.perf_counter()
    first = run_pose_model(model, image, confidence, image_size, device)
    first_ms = (time.perf_counter() - first_start) * 1000.0

    def fallback(reason: str, roi=None) -> TwoStageResult:
        return TwoStageResult(
            first,
            first,
            False,
            reason,
            roi,
            first_ms,
            None,
            (time.perf_counter() - total_start) * 1000.0,
        )

    if first.box_xyxy is None or not first.has_keypoints:
        return fallback("first_detection_missing")
    if first.box_confidence < minimum_first_box_confidence:
        return fallback("first_box_low_confidence")
    roi = expand_roi(first.box_xyxy, width, height, roi_margin, minimum_roi_side)
    if roi is None:
        return fallback("invalid_roi")
    left, top, right, bottom = roi
    if ((right - left) * (bottom - top)) / (width * height) >= maximum_roi_area_fraction:
        return fallback("roi_near_full_frame", roi)
    crop = np.ascontiguousarray(image[top:bottom, left:right])
    second_start = time.perf_counter()
    refinement_model = model if second_model is None else second_model
    second = run_pose_model(refinement_model, crop, confidence, image_size, device)
    second_ms = (time.perf_counter() - second_start) * 1000.0
    if second.box_xyxy is None or not second.has_keypoints:
        return TwoStageResult(
            first,
            first,
            False,
            "second_detection_missing",
            roi,
            first_ms,
            second_ms,
            (time.perf_counter() - total_start) * 1000.0,
        )
    final = remap_prediction(second, roi, width, height)
    return TwoStageResult(
        first,
        final,
        True,
        None,
        roi,
        first_ms,
        second_ms,
        (time.perf_counter() - total_start) * 1000.0,
    )


def prediction_to_dict(prediction: PosePrediction) -> dict:
    return {
        "image_width": prediction.image_width,
        "image_height": prediction.image_height,
        "box_xyxy": prediction.box_xyxy,
        "box_confidence": prediction.box_confidence,
        "keypoints_xy": prediction.keypoints_xy,
        "keypoint_confidences": prediction.keypoint_confidences,
        "keypoints_normalized": prediction.normalized_keypoints(),
    }


def result_to_dict(result: TwoStageResult) -> dict:
    return {
        "used_second_stage": result.used_second_stage,
        "fallback_reason": result.fallback_reason,
        "roi_xyxy": result.roi_xyxy,
        "timing_ms": {
            "first_inference": result.first_inference_ms,
            "second_inference": result.second_inference_ms,
            "total_inference": result.total_inference_ms,
        },
        "first": prediction_to_dict(result.first),
        "final": prediction_to_dict(result.final),
    }
