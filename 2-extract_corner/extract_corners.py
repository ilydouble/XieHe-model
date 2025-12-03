#!/usr/bin/env python3
"""
从分割数据集中提取每个目标的四个角点（最小外接矩形的顶点）
生成新的关键点检测数据集

使用方法:
    python extract_corners.py --seg_dir ../seg_data --output_dir ../pose_corner_data --visualize
"""

import os
import argparse
import shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import yaml


def parse_seg_label(label_path: Path) -> list:
    """
    解析分割标签文件
    格式: class_id x_c y_c w h poly_x1 poly_y1 poly_x2 poly_y2 ...
    返回: [(class_id, polygon_points), ...]
    """
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # 至少需要 class + bbox(4) + 1个点(2)
                continue

            class_id = int(parts[0])
            # 跳过前5个值(class, x_c, y_c, w, h)，剩下的是多边形坐标
            coords = list(map(float, parts[5:]))

            # 坐标是成对的 (x, y)
            if len(coords) >= 6:  # 至少3个点才能形成多边形
                points = np.array(coords).reshape(-1, 2)
                objects.append((class_id, points))

    return objects


def get_corner_points(polygon: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    计算多边形的最小外接矩形的四个顶点

    Args:
        polygon: 归一化的多边形坐标 (N, 2)
        img_w, img_h: 图像尺寸

    Returns:
        corners: 四个顶点的归一化坐标 (4, 2)，按顺序排列（左上、右上、右下、左下）
    """
    # 转换为像素坐标
    pts = polygon.copy()
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    pts = pts.astype(np.float32)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)  # 获取4个顶点

    # 对顶点排序：左上、右上、右下、左下
    box = sort_corners(box)

    # 转回归一化坐标
    box[:, 0] /= img_w
    box[:, 1] /= img_h

    return box


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """
    将四个角点按照 左上、右上、右下、左下 的顺序排列
    """
    # 计算中心点
    center = corners.mean(axis=0)

    # 按照与中心点的角度排序
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])

    # 找到左上角（角度最接近 -135度 或 -3π/4）
    sorted_indices = np.argsort(angles)

    # 重新排序：从左上开始顺时针
    # 先找y值最小的两个点（上边的两个点）
    top_indices = np.argsort(corners[:, 1])[:2]
    bottom_indices = np.argsort(corners[:, 1])[2:]

    # 上边两个点按x排序（左、右）
    top_left_idx = top_indices[np.argmin(corners[top_indices, 0])]
    top_right_idx = top_indices[np.argmax(corners[top_indices, 0])]

    # 下边两个点按x排序（左、右）
    bottom_left_idx = bottom_indices[np.argmin(corners[bottom_indices, 0])]
    bottom_right_idx = bottom_indices[np.argmax(corners[bottom_indices, 0])]

    # 按照 左上、右上、右下、左下 顺序
    sorted_corners = corners[[top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx]]

    return sorted_corners


def create_pose_label(objects: list, img_w: int, img_h: int) -> list:
    """
    将分割对象转换为关键点标注格式

    YOLO Pose格式: class x_c y_c w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    """
    pose_labels = []

    for class_id, polygon in objects:
        # 获取四个角点
        corners = get_corner_points(polygon, img_w, img_h)

        # 计算bbox (从多边形计算)
        pts = polygon.copy()
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        # 构建标签行
        label_parts = [str(class_id), f"{x_c:.6f}", f"{y_c:.6f}", f"{w:.6f}", f"{h:.6f}"]

        # 添加4个关键点 (x, y, visibility=2表示可见)
        for i in range(4):
            kp_x, kp_y = corners[i]
            label_parts.extend([f"{kp_x:.6f}", f"{kp_y:.6f}", "2"])

        pose_labels.append(" ".join(label_parts))

    return pose_labels


def visualize_corners(img_path: Path, objects: list, output_path: Path):
    """
    可视化提取的角点
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return

    h, w = img.shape[:2]

    colors = [
        (0, 255, 0),    # 绿色 - 左上
        (255, 0, 0),    # 蓝色 - 右上
        (0, 0, 255),    # 红色 - 右下
        (255, 255, 0),  # 青色 - 左下
    ]
    corner_names = ['TL', 'TR', 'BR', 'BL']

    for class_id, polygon in objects:
        corners = get_corner_points(polygon, w, h)

        # 转换为像素坐标
        corners_px = corners.copy()
        corners_px[:, 0] *= w
        corners_px[:, 1] *= h
        corners_px = corners_px.astype(np.int32)

        # 画原始多边形轮廓
        poly_px = polygon.copy()
        poly_px[:, 0] *= w
        poly_px[:, 1] *= h
        poly_px = poly_px.astype(np.int32)
        cv2.polylines(img, [poly_px], True, (128, 128, 128), 1)

        # 画最小外接矩形
        cv2.polylines(img, [corners_px], True, (0, 255, 255), 2)

        # 画四个角点
        for i, (cx, cy) in enumerate(corners_px):
            cv2.circle(img, (cx, cy), 6, colors[i], -1)
            cv2.putText(img, f"{class_id}_{corner_names[i]}",
                       (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

    cv2.imwrite(str(output_path), img)


def process_dataset(seg_dir: Path, output_dir: Path, visualize: bool = False):
    """
    处理整个数据集
    """
    # 创建输出目录结构
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'test').mkdir(parents=True, exist_ok=True)

    if visualize:
        (output_dir / 'visualize').mkdir(parents=True, exist_ok=True)

    # 处理每个split
    for split in ['train', 'val', 'test']:
        images_dir = seg_dir / 'images' / split
        labels_dir = seg_dir / 'labels' / split

        if not images_dir.exists():
            print(f"⚠️  {split} 目录不存在，跳过")
            continue

        # 获取所有图像
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in images_dir.iterdir()
                       if f.suffix.lower() in image_extensions]

        # 统计
        processed = 0
        skipped_no_label = 0
        skipped_empty = 0

        print(f"\n📂 处理 {split} 集 ({len(image_files)} 张图像)...")

        for img_path in tqdm(image_files, desc=f"  {split}"):
            # 先检查标签是否存在
            label_path = labels_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                skipped_no_label += 1
                continue

            # 解析标签
            objects = parse_seg_label(label_path)
            if not objects:
                skipped_empty += 1
                continue

            # 读取图像尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # 生成pose标签
            pose_labels = create_pose_label(objects, w, h)

            # 只有成功生成标签才复制图像
            shutil.copy(img_path, output_dir / 'images' / split / img_path.name)

            # 保存标签
            output_label_path = output_dir / 'labels' / split / (img_path.stem + '.txt')
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(pose_labels))

            processed += 1

            # 可视化
            if visualize:
                vis_path = output_dir / 'visualize' / f"{split}_{img_path.stem}.jpg"
                visualize_corners(img_path, objects, vis_path)

        # 打印统计
        if skipped_no_label > 0 or skipped_empty > 0:
            print(f"    ✅ 处理: {processed}, ⚠️ 跳过(无标签): {skipped_no_label}, ⚠️ 跳过(空标签): {skipped_empty}")


def create_dataset_yaml(output_dir: Path, class_names: list):
    """
    创建 dataset.yaml 配置文件
    """
    # 生成关键点名称: class_name_1, class_name_2, class_name_3, class_name_4
    kpt_names = []
    for name in class_names:
        kpt_names.extend([f"{name}_1", f"{name}_2", f"{name}_3", f"{name}_4"])

    config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names,
        'kpt_shape': [4, 3],  # 4个关键点，每个有 x, y, visibility
        'flip_idx': [1, 0, 3, 2],  # 左右翻转时的关键点映射 (TL<->TR, BL<->BR)
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        f.write(f"\n# Keypoint names (for reference):\n")
        f.write(f"# 0: Corner_1 (Top-Left)\n")
        f.write(f"# 1: Corner_2 (Top-Right)\n")
        f.write(f"# 2: Corner_3 (Bottom-Right)\n")
        f.write(f"# 3: Corner_4 (Bottom-Left)\n")

    print(f"\n📄 生成配置文件: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='从分割数据集提取角点，生成关键点数据集')
    parser.add_argument('--seg_dir', type=str, required=True,
                        help='分割数据集目录 (包含images和labels文件夹)')
    parser.add_argument('--output_dir', type=str, default='../pose_corner_data',
                        help='输出目录 (默认: ../pose_corner_data)')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化结果')

    args = parser.parse_args()

    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("分割数据集 → 关键点数据集 转换工具")
    print("=" * 60)
    print(f"输入目录: {seg_dir}")
    print(f"输出目录: {output_dir}")
    print(f"可视化: {'是' if args.visualize else '否'}")

    # 定义类别名称 (18个椎骨)
    # 根据实际情况修改
    class_names = [f"V{i}" for i in range(18)]  # V0, V1, ..., V17

    # 处理数据集
    process_dataset(seg_dir, output_dir, args.visualize)

    # 创建yaml配置
    create_dataset_yaml(output_dir, class_names)

    print("\n" + "=" * 60)
    print("✅ 转换完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 图像: {output_dir}/images/{{train,val,test}}/")
    print(f"  - 标签: {output_dir}/labels/{{train,val,test}}/")
    print(f"  - 配置: {output_dir}/dataset.yaml")
    if args.visualize:
        print(f"  - 可视化: {output_dir}/visualize/")


if __name__ == '__main__':
    main()

