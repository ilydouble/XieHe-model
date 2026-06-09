#!/usr/bin/env python3
"""
将 AP 正面脊柱 Pose 标签转换为 Detection 标签

输入: datasets/pose_data  (YOLO Pose 格式, 1类 spine, 6 关键点)
输出: datasets/pose_det_data (YOLO Detect 格式, 6 类, 每类=一个解剖标志点小 bbox)

关键点 → 类别映射:
  kp0 CR → class 0   右侧锁骨最高点
  kp1 CL → class 1   左侧锁骨最高点
  kp2 IR → class 2   右侧髂骨最高点
  kp3 IL → class 3   左侧髂骨最高点
  kp4 SR → class 4   骶一上终板右缘点
  kp5 SL → class 5   骶一上终板左缘点

用法:
  python convert_pose_to_detection.py [--bbox-ratio 0.04] [--input ...] [--output ...]
"""

import argparse
import shutil
from pathlib import Path


KPT_NAMES = ['CR', 'CL', 'IR', 'IL', 'SR', 'SL']
SPLITS = ['train', 'val', 'test']


def parse_args():
    p = argparse.ArgumentParser(description='Pose → Detection label converter')
    p.add_argument('--input',  default='../datasets/pose_data',
                   help='输入 pose 数据集根目录 (默认: ../datasets/pose_data)')
    p.add_argument('--output', default='../datasets/pose_det_data',
                   help='输出 detection 数据集根目录 (默认: ../datasets/pose_det_data)')
    p.add_argument('--bbox-ratio', type=float, default=0.04,
                   help='每个关键点 bbox 边长占图像短边的比例 (默认: 0.04 = 4%%)')
    p.add_argument('--dry-run', action='store_true',
                   help='只统计，不写文件')
    return p.parse_args()


def convert_label(label_path: Path, bbox_ratio: float) -> tuple[list[str], int]:
    """
    读取一个 pose label 文件，返回 (detection_lines, skip_count)。

    Pose 格式 (每行):
      class cx cy w h  kp0x kp0y vis0  kp1x kp1y vis1  ...  kp5x kp5y vis5

    处理规则:
      - 只处理 class=0 (spine 六关键点) 的行
      - 根据实际 token 数量动态确定可用关键点数 (不强制要求 23 个)
      - 跳过 vis=0 的不可见关键点

    Detection 输出 (每关键点一行):
      class_id  cx  cy  bw  bh
    """
    lines_out = []
    skipped_rows = 0
    bw = bbox_ratio
    bh = bbox_ratio

    with open(label_path) as f:
        for raw in f:
            parts = raw.strip().split()
            # 最少需要: class(1) + bbox(4) + 至少一个完整关键点(3) = 8 tokens
            if len(parts) < 8:
                skipped_rows += 1
                continue

            pose_class = int(float(parts[0]))
            if pose_class != 0:
                # class != 0 的行不是标准六关键点 spine 标注，跳过
                skipped_rows += 1
                continue

            # 动态计算可用关键点数量
            n_kpts = (len(parts) - 5) // 3   # 5 = class + 4 bbox tokens
            n_kpts = min(n_kpts, 6)           # 最多取 6 个

            for kp_idx in range(n_kpts):
                base = 5 + kp_idx * 3
                kx  = float(parts[base])
                ky  = float(parts[base + 1])
                vis = int(float(parts[base + 2]))
                if vis == 0:
                    continue
                cx = max(bw / 2, min(1 - bw / 2, kx))
                cy = max(bh / 2, min(1 - bh / 2, ky))
                lines_out.append(
                    f'{kp_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'
                )

    return lines_out, skipped_rows


def process_split(src_root: Path, dst_root: Path,
                  split: str, bbox_ratio: float, dry_run: bool) -> dict:
    src_img_dir = src_root / 'images' / split
    src_lbl_dir = src_root / 'labels' / split
    dst_img_dir = dst_root / 'images' / split
    dst_lbl_dir = dst_root / 'labels' / split

    if not src_lbl_dir.exists():
        return {'images': 0, 'labels': 0, 'detections': 0, 'skipped': 0}

    if not dry_run:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {'images': 0, 'labels': 0, 'detections': 0,
             'skipped_files': 0, 'skipped_rows': 0}

    for lbl_file in sorted(src_lbl_dir.glob('*.txt')):
        det_lines, skip_rows = convert_label(lbl_file, bbox_ratio)
        stats['skipped_rows'] += skip_rows

        if not det_lines:
            stats['skipped_files'] += 1
            continue

        if not dry_run:
            # 复制图像
            for ext in ('.png', '.jpg', '.jpeg'):
                img_src = src_img_dir / (lbl_file.stem + ext)
                if img_src.exists():
                    shutil.copy2(img_src, dst_img_dir / img_src.name)
                    stats['images'] += 1
                    break
            # 写 detection label
            dst_lbl = dst_lbl_dir / lbl_file.name
            dst_lbl.write_text('\n'.join(det_lines) + '\n')

        stats['labels'] += 1
        stats['detections'] += len(det_lines)

    return stats


def write_data_yaml(dst_root: Path):
    yaml_content = f"""# AP 正面脊柱解剖标志点检测数据集
# YOLO Detect 格式: 6 类，每类对应一个解剖标志点

path: {dst_root.resolve()}
train: images/train
val:   images/val
test:  images/test

# 类别数
nc: 6

# 类别名称 (与原 Pose 关键点顺序一致)
names:
  0: CR   # 右侧锁骨最高点
  1: CL   # 左侧锁骨最高点
  2: IR   # 右侧髂骨最高点
  3: IL   # 左侧髂骨最高点
  4: SR   # 骶一上终板右缘点
  5: SL   # 骶一上终板左缘点
"""
    (dst_root / 'data.yaml').write_text(yaml_content)
    print(f'  ✅ 写入 {dst_root / "data.yaml"}')


def main():
    args = parse_args()
    src_root = Path(args.input)
    dst_root = Path(args.output)

    print('=' * 65)
    print('🔄  Pose → Detection 标签转换')
    print('=' * 65)
    print(f'  输入目录  : {src_root}')
    print(f'  输出目录  : {dst_root}')
    print(f'  bbox 比例 : {args.bbox_ratio * 100:.1f}%  '
          f'(800px 图像 ≈ {int(args.bbox_ratio * 800)}px)')
    print(f'  dry-run   : {args.dry_run}')
    print()

    if not src_root.exists():
        raise FileNotFoundError(f'输入目录不存在: {src_root}')

    total = {'images': 0, 'labels': 0, 'detections': 0,
             'skipped_files': 0, 'skipped_rows': 0}

    for split in SPLITS:
        stats = process_split(src_root, dst_root, split, args.bbox_ratio, args.dry_run)
        any_data = stats['labels'] > 0 or stats['skipped_files'] > 0
        if any_data:
            print(f'  [{split:5s}]  图像={stats["images"]}  '
                  f'标签={stats["labels"]}  '
                  f'检测框={stats["detections"]}  '
                  f'跳过文件={stats["skipped_files"]}  '
                  f'跳过行={stats["skipped_rows"]}')
        for k in total:
            total[k] += stats[k]

    if not args.dry_run:
        write_data_yaml(dst_root)

    print()
    print('=' * 65)
    print('✅  转换完成！')
    print(f'   总图像    : {total["images"]}')
    print(f'   总标签    : {total["labels"]}')
    print(f'   总检测框  : {total["detections"]}  '
          f'(平均 {total["detections"] / max(total["labels"], 1):.1f} 个/图)')
    print(f'   跳过文件  : {total["skipped_files"]}  '
          f'(全部关键点不可见或非 class=0)')
    print(f'   跳过行    : {total["skipped_rows"]}  '
          f'(class!=0 或 token 数不足)')
    if not args.dry_run:
        print(f'   数据集    : {dst_root}')
    print('=' * 65)


if __name__ == '__main__':
    main()
