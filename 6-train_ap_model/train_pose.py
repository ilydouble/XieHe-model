#!/usr/bin/env python3
"""
AP 正面脊柱六关键点检测模型训练
- 模型: YOLO11 Pose
- 任务: 1 类 (spine), 6 个关键点 (CR/CL/IR/IL/SR/SL)
- 数据: datasets/pose_data (train/val/test)

用法:
    python train_pose.py [--model n|s|m|l] [--epochs N] [--batch N]
                         [--imgsz N] [--device 0] [--name NAME] [--resume]
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train AP 6-Keypoint Pose Model')
    parser.add_argument('--model',   default='s',     choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO11 model size (default: s)')
    parser.add_argument('--epochs',  type=int, default=200,
                        help='Training epochs (default: 200)')
    parser.add_argument('--batch',   type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--imgsz',   type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--device',  default='0',
                        help='CUDA device id or "cpu" (default: 0)')
    parser.add_argument('--name',    default='train',
                        help='Experiment name (default: train)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Dataloader workers (default: 8)')
    parser.add_argument('--data', default='pose_data.yaml',
                        help='Dataset YAML path or filename relative to this script')
    parser.add_argument('--augmentation-profile', default='standard',
                        choices=['standard', 'roi_low'],
                        help='Augmentation preset: existing standard or controlled ROI experiment')
    parser.add_argument('--resume',  action='store_true',
                        help='Resume from last checkpoint')
    return parser.parse_args(argv)


def resolve_data_yaml(value, script_dir):
    requested = Path(value).expanduser()
    candidates = [requested] if requested.is_absolute() else [script_dir / requested, requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'数据集配置文件不存在: {value}')


def augmentation_config(profile):
    common = {
        'rect': False,
        'flipud': 0.0,
        'fliplr': 0.5,
        'dropout': 0.1,
    }
    if profile == 'standard':
        return {
            **common,
            'multi_scale': True,
            'degrees': 5.0,
            'translate': 0.1,
            'scale': 0.5,
            'perspective': 0.0001,
            'hsv_h': 0.01,
            'hsv_s': 0.3,
            'hsv_v': 0.2,
            'mosaic': 1.0,
            'close_mosaic': 30,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'erasing': 0.3,
            'auto_augment': 'randaugment',
        }
    if profile == 'roi_low':
        return {
            **common,
            'multi_scale': False,
            'degrees': 3.0,
            'translate': 0.05,
            'scale': 0.15,
            'perspective': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.1,
            'hsv_v': 0.15,
            'mosaic': 0.0,
            'close_mosaic': 0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'erasing': 0.0,
            'auto_augment': None,
        }
    raise ValueError(f'未知增强预设: {profile}')


def main():
    args = parse_args()

    from ultralytics import YOLO

    script_dir  = Path(__file__).parent
    data_yaml   = resolve_data_yaml(args.data, script_dir)
    weights_dir = script_dir.parent / 'weights'
    pretrained  = weights_dir / f'yolo11{args.model}-pose.pt'

    print('=' * 70)
    print('🚀  AP 正面模型训练 — 六关键点姿态检测 (Pose)')
    print('    关键点: CR / CL / IR / IL / SR / SL')
    print('=' * 70)
    print(f'  模型大小 : YOLO11{args.model}-pose')
    print(f'  预训练权重: {pretrained}')
    print(f'  训练轮数 : {args.epochs}')
    print(f'  批次大小 : {args.batch}')
    print(f'  图像大小 : {args.imgsz}')
    print(f'  设备    : {args.device}')
    print(f'  实验名称 : {args.name}')
    print(f'  数据集   : {data_yaml}')
    print(f'  增强预设 : {args.augmentation_profile}')
    print('=' * 70)
    print()

    # 加载模型
    if args.resume:
        ckpt = script_dir / f'runs/pose/{args.name}/weights/last.pt'
        if ckpt.exists():
            print(f'从检查点继续训练: {ckpt}')
            model = YOLO(str(ckpt))
        else:
            print(f'⚠️  未找到检查点，从头开始')
            model = YOLO(str(pretrained) if pretrained.exists()
                         else f'yolo11{args.model}-pose.pt')
    else:
        if pretrained.exists():
            print(f'加载预训练权重: {pretrained}')
            model = YOLO(str(pretrained))
        else:
            print(f'⚠️  未找到本地权重，尝试从 ultralytics 下载')
            model = YOLO(f'yolo11{args.model}-pose.pt')

    # YAML中的path相对训练脚本目录，保证从仓库根目录直接调用时也能正确解析。
    previous_cwd = Path.cwd()
    os.chdir(script_dir)
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            amp=False,        # 关闭 AMP 自动检查，避免因 yolo11n.pt 下载失败而崩溃
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.001,
            warmup_epochs=3.0,
            **augmentation_config(args.augmentation_profile),
            # 损失权重 (关键点检测更重视 pose loss)
            box=7.5,
            cls=0.5,
            pose=12.0,
            kobj=2.0,
            # 输出
            project=str(script_dir / 'runs/pose'),
            name=args.name,
            plots=True,
            verbose=True,
        )
    finally:
        os.chdir(previous_cwd)

    print()
    print('=' * 70)
    print('✅  训练完成！')
    print(f'   最佳模型: runs/pose/{args.name}/weights/best.pt')
    print('=' * 70)


if __name__ == '__main__':
    main()
