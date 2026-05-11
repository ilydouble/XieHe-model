#!/usr/bin/env python3
"""
AP 正面脊柱椎体角点检测模型训练
- 模型: YOLO11 Pose
- 任务: 18 个椎体类别, 每个 4 个角点关键点 (TL/TR/BR/BL)
- 数据: datasets/pose_corner_data (train/val/test)

用法:
    python train_corner.py [--model n|s|m|l] [--epochs N] [--batch N]
                           [--imgsz N] [--device 0] [--name NAME] [--resume]
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train AP Corner Detection Model')
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
    parser.add_argument('--resume',  action='store_true',
                        help='Resume from last checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    script_dir = Path(__file__).parent
    data_yaml  = script_dir / 'corner_data.yaml'
    weights_dir = script_dir.parent / 'weights'
    pretrained  = weights_dir / f'yolo11{args.model}-pose.pt'

    print('=' * 70)
    print('🚀  AP 正面模型训练 — 椎体角点检测 (Corner)')
    print('=' * 70)
    print(f'  模型大小 : YOLO11{args.model}-pose')
    print(f'  预训练权重: {pretrained}')
    print(f'  训练轮数 : {args.epochs}')
    print(f'  批次大小 : {args.batch}')
    print(f'  图像大小 : {args.imgsz}')
    print(f'  设备    : {args.device}')
    print(f'  实验名称 : {args.name}')
    print(f'  数据集   : {data_yaml}')
    print('=' * 70)
    print()

    if not data_yaml.exists():
        raise FileNotFoundError(f'数据集配置文件不存在: {data_yaml}')

    # 加载模型
    if args.resume:
        ckpt = script_dir / f'runs/corner/{args.name}/weights/last.pt'
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

    # 开始训练
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
        weight_decay=0.001,   # 从 0.0005 加大，抑制过拟合
        warmup_epochs=3.0,
        # 尺寸泛化：开启 multi_scale，训练时随机缩放 0.5x~1.5x imgsz
        # 注意：最大会到 1.5x imgsz，显存需求大，建议 imgsz<=800 或减小 batch
        multi_scale=True,
        rect=False,           # 正方形 padding，不用矩形 batch（默认 False）
        # 数据增强 (正面脊柱不上下翻转)
        flipud=0.0,
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,            # 从 0.3 加大，增强缩放鲁棒性
        perspective=0.0001,   # 轻微透视变换，模拟拍摄角度差异
        # 颜色/亮度增强（应对不同拍摄条件，X光片用保守值）
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        # 混合/遮挡增强
        mosaic=1.0,
        close_mosaic=30,      # 从默认 10 延长，减少最后阶段的过拟合
        mixup=0.1,            # 轻微 mixup 增强泛化
        copy_paste=0.1,       # 拼贴增强，增加位置/尺度多样性
        erasing=0.3,          # 随机擦除，防止过拟合
        dropout=0.1,          # 启用 dropout 正则化
        auto_augment='randaugment',  # 自动增强策略
        # 损失权重
        box=7.5,
        cls=0.5,
        pose=12.0,
        kobj=1.0,
        # 输出
        project=str(script_dir / 'runs/corner'),
        name=args.name,
        plots=True,
        verbose=True,
    )

    print()
    print('=' * 70)
    print('✅  训练完成！')
    print(f'   最佳模型: runs/corner/{args.name}/weights/best.pt')
    print('=' * 70)


if __name__ == '__main__':
    main()
