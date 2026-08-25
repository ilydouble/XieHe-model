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
BLACK_THRESHOLD = 12   # 灰度均值低于此值的行/列视为黑边
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


def crop_black_border(img_bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    裁剪图像四边的黑色边框。
    返回 (裁剪后图像, (x1, y1, x2, y2)) — 坐标用于反映射回原图。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    col_means = gray.mean(axis=0)   # 每列均值
    row_means = gray.mean(axis=1)   # 每行均值

    left   = next((c for c in range(w)       if col_means[c] > BLACK_THRESHOLD), 0)
    right  = next((c for c in range(w-1,-1,-1) if col_means[c] > BLACK_THRESHOLD), w-1) + 1
    top    = next((r for r in range(h)       if row_means[r] > BLACK_THRESHOLD), 0)
    bottom = next((r for r in range(h-1,-1,-1) if row_means[r] > BLACK_THRESHOLD), h-1) + 1

    # 加少量 padding 防止裁剪过紧
    pad = 4
    left   = max(0, left   - pad)
    top    = max(0, top    - pad)
    right  = min(w, right  + pad)
    bottom = min(h, bottom + pad)

    cropped = img_bgr[top:bottom, left:right]
    return cropped, (left, top, right, bottom)


def preprocess_image(img_path: str) -> tuple[np.ndarray, dict]:
    """
    读取图像并做预处理：
      1. 转 RGB（处理 RGBA）
      2. 裁剪黑边
      3. 记录变换参数，供坐标反映射使用
    返回 (预处理后的 BGR 图像, 变换元数据)
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, {}

    # 处理 RGBA
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    orig_h, orig_w = img.shape[:2]
    cropped, (x1, y1, x2, y2) = crop_black_border(img)
    crop_h, crop_w = cropped.shape[:2]

    border_removed = (x1 > 2 or y1 > 2 or x2 < orig_w - 2 or y2 < orig_h - 2)

    meta = {
        'orig_size':   (orig_w, orig_h),
        'crop_box':    (x1, y1, x2, y2),   # 裁剪区域在原图中的位置
        'crop_size':   (crop_w, crop_h),
        'border_removed': border_removed,
    }
    return cropped, meta


def remap_points(points: dict, meta: dict) -> dict:
    """将裁剪图上的归一化坐标反映射回原图的归一化坐标"""
    if not points or not meta:
        return points
    orig_w, orig_h = meta['orig_size']
    x1, y1, x2, y2 = meta['crop_box']
    crop_w, crop_h = meta['crop_size']

    remapped = {}
    for cls_id, (cx, cy) in points.items():
        # 裁剪图归一化 → 裁剪图像素
        px = cx * crop_w + x1
        py = cy * crop_h + y1
        # 裁剪图像素 → 原图归一化
        remapped[cls_id] = (px / orig_w, py / orig_h)
    return remapped


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


def run_pose(model, img_bgr: np.ndarray, conf: float) -> dict:
    """运行 Pose 模型，输入已预处理的 BGR 图像，返回 {class_id: (cx_norm, cy_norm)}"""
    results = model(img_bgr, conf=conf, verbose=False)
    points = {}
    for r in results:
        if r.keypoints is None:
            continue
        kpts = r.keypoints.xyn
        if kpts is None or len(kpts) == 0:
            continue
        best = 0
        if r.boxes is not None and len(r.boxes) > 1:
            best = int(r.boxes.conf.argmax())
        for kp_idx in range(min(6, kpts.shape[1])):
            x, y = float(kpts[best, kp_idx, 0]), float(kpts[best, kp_idx, 1])
            if x > 0 or y > 0:
                points[kp_idx] = (x, y)
    return points


def run_detect(model, img_bgr: np.ndarray, conf: float) -> dict:
    """运行 Detection 模型，输入已预处理的 BGR 图像，返回 {class_id: (cx_norm, cy_norm)}"""
    results = model(img_bgr, conf=conf, verbose=False)
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
            if cls_id not in points or float(box.conf[0]) > points[cls_id][2]:
                points[cls_id] = (cx, cy, float(box.conf[0]))
    return {k: (v[0], v[1]) for k, v in points.items()}


def make_panel(orig_img: np.ndarray, pose_pts: dict, det_pts: dict,
               meta: dict, target_h: int = 900) -> np.ndarray:
    """在原始图像（未裁剪）上绘制坐标已反映射的关键点"""
    img = orig_img.copy()
    h, w = img.shape[:2]
    scale = target_h / h
    img = cv2.resize(img, (int(w * scale), target_h))

    # 若做过黑边裁剪，在图上标注提示
    if meta.get('border_removed'):
        x1, y1, x2, y2 = meta['crop_box']
        sx = int(x1 * scale); sy = int(y1 * scale)
        ex = int(x2 * scale); ey = int(y2 * scale)
        cv2.rectangle(img, (sx, sy), (ex, ey), (0, 200, 255), 2)

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

        # ── 预处理：读图 + 裁黑边 ──────────────────────────
        orig_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if orig_img is None:
            print(f'    ⚠️  无法读取，跳过'); continue
        if len(orig_img.shape) == 3 and orig_img.shape[2] == 4:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)

        cropped, meta = preprocess_image(str(img_path))
        oh, ow = orig_img.shape[:2]
        ch, cw = cropped.shape[:2]
        border_info = (f'黑边裁剪 {ow}x{oh}→{cw}x{ch}'
                       if meta.get('border_removed') else f'{ow}x{oh}')
        print(f'    {border_info}')

        # ── 推理（在裁剪图上） ────────────────────────────
        pose_pts_crop = run_pose(pose_model, cropped, args.conf)
        det_pts_crop  = run_detect(det_model, cropped, args.conf)

        # ── 坐标反映射回原图 ──────────────────────────────
        pose_pts = remap_points(pose_pts_crop, meta)
        det_pts  = remap_points(det_pts_crop,  meta)

        panel = make_panel(orig_img, pose_pts, det_pts, meta)

        out_path = output_dir / f'cmp_{img_path.stem}.jpg'
        cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'    pose={len(pose_pts)}/6  detect={len(det_pts)}/6  → {out_path.name}')

    print(f'\n✅ 完成！结果保存在: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
