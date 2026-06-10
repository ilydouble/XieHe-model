#!/usr/bin/env python3
"""
Pose 模型 vs Detection 模型 六关键点定位对比可视化

用法:
  python compare_pose_vs_detect.py [--image-dir ...] [--pose-model ...] [--det-model ...]
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

KPT_NAMES  = ['CR', 'CL', 'IR', 'IL', 'SR', 'SL']
# 每个关键点的颜色 BGR
KPT_COLORS = [
    (255, 100,  50),   # CR  蓝
    (255, 180,  50),   # CL  天蓝
    ( 50, 200,  50),   # IR  绿
    ( 50, 200, 150),   # IL  青绿
    ( 50,  80, 255),   # SR  红
    ( 80, 150, 255),   # SL  橙红
]
IMG_SUFFIXES = {'.png', '.jpg', '.jpeg'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir',   default='medical-image-files')
    p.add_argument('--pose-model',
                   default='../6-train_ap_model/runs/pose/best_performance2/weights/best.pt')
    p.add_argument('--det-model',
                   default='../6-train_ap_model/runs/detect/best_performance-3/weights/best.pt')
    p.add_argument('--output-dir',  default='compare_results')
    p.add_argument('--conf',        type=float, default=0.25)
    p.add_argument('--keyword',     default='正位',
                   help='只处理文件名含此关键词的图像')
    return p.parse_args()


def draw_landmarks(img: np.ndarray, points: dict, title: str) -> np.ndarray:
    """在图像上绘制关键点，points = {class_id: (cx, cy)} 或 None"""
    out = img.copy()
    h, w = out.shape[:2]

    # 标题
    cv2.rectangle(out, (0, 0), (w, 42), (30, 30, 30), -1)
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)

    if not points:
        cv2.putText(out, 'No detections', (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return out

    for cls_id, (cx, cy) in points.items():
        color = KPT_COLORS[cls_id]
        name  = KPT_NAMES[cls_id]
        px, py = int(cx * w), int(cy * h)

        # 圆点
        cv2.circle(out, (px, py), 10, color, -1)
        cv2.circle(out, (px, py), 12, (255, 255, 255), 2)

        # 标签背景
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (px + 14, py - th - 4), (px + 14 + tw + 4, py + 4),
                      color, -1)
        cv2.putText(out, name, (px + 16, py), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    return out


def run_pose(model, img_path: str, conf: float) -> dict:
    """运行 Pose 模型，返回 {class_id: (cx_norm, cy_norm)}"""
    results = model(img_path, conf=conf, verbose=False)
    points = {}
    for r in results:
        if r.keypoints is None:
            continue
        kpts = r.keypoints.xyn  # shape [N, 6, 2], normalized
        if kpts is None or len(kpts) == 0:
            continue
        # 取置信度最高的检测框
        best = 0
        if r.boxes is not None and len(r.boxes) > 1:
            best = int(r.boxes.conf.argmax())
        for kp_idx in range(min(6, kpts.shape[1])):
            x, y = float(kpts[best, kp_idx, 0]), float(kpts[best, kp_idx, 1])
            if x > 0 or y > 0:
                points[kp_idx] = (x, y)
    return points


def run_detect(model, img_path: str, conf: float) -> dict:
    """运行 Detection 模型，返回 {class_id: (cx_norm, cy_norm)}"""
    results = model(img_path, conf=conf, verbose=False)
    points = {}
    for r in results:
        if r.boxes is None:
            continue
        h, w = r.orig_shape
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            # 同一类别取置信度最高的
            if cls_id not in points or float(box.conf[0]) > points[cls_id][2]:
                points[cls_id] = (cx, cy, float(box.conf[0]))
    return {k: (v[0], v[1]) for k, v in points.items()}


def make_panel(img_path: Path, pose_pts: dict, det_pts: dict,
               target_h: int = 900) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = target_h / h
    img = cv2.resize(img, (int(w * scale), target_h))

    left  = draw_landmarks(img, pose_pts,  'Pose Model')
    right = draw_landmarks(img, det_pts,   'Detect Model')

    # 分隔线
    sep = np.full((target_h, 6, 3), 200, dtype=np.uint8)
    panel = np.hstack([left, sep, right])

    # 底部图例
    legend_h = 40
    legend = np.full((legend_h, panel.shape[1], 3), 50, dtype=np.uint8)
    x_off = 10
    for i, (name, color) in enumerate(zip(KPT_NAMES, KPT_COLORS)):
        cv2.circle(legend, (x_off + 8, 20), 7, color, -1)
        cv2.putText(legend, name, (x_off + 20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        x_off += 80
    return np.vstack([panel, legend])


def main():
    args = parse_args()
    from ultralytics import YOLO

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('加载模型...')
    pose_model = YOLO(args.pose_model)
    det_model  = YOLO(args.det_model)

    images = [f for f in sorted(image_dir.iterdir())
              if f.suffix.lower() in IMG_SUFFIXES
              and (not args.keyword or args.keyword in f.name)]

    print(f'找到 {len(images)} 张{args.keyword}图像，开始推理...\n')

    for img_path in images:
        print(f'  处理: {img_path.name}')
        pose_pts = run_pose(pose_model, str(img_path), args.conf)
        det_pts  = run_detect(det_model, str(img_path), args.conf)

        panel = make_panel(img_path, pose_pts, det_pts)
        if panel is None:
            print(f'    ⚠️  无法读取图像，跳过')
            continue

        out_path = output_dir / f'cmp_{img_path.stem}.jpg'
        cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'    pose={len(pose_pts)}/6  detect={len(det_pts)}/6  → {out_path.name}')

    print(f'\n✅ 完成！结果保存在: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
