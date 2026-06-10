#!/usr/bin/env python3
"""
AP 正面脊柱解剖标志点检测模型训练
- 模型  : YOLO11 Detect
- 任务  : 6 类目标检测 (CR / CL / IR / IL / SR / SL)
- 数据  : datasets/pose_det_data (由 convert_pose_to_detection.py 生成)
- 推理  : 取检测框中心点作为解剖标志点坐标

用法:
    python train_detection.py [--model n|s|m|l] [--epochs N] [--batch N]
                              [--imgsz N] [--device 0] [--name NAME] [--resume]
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser(description='Train AP Landmark Detection Model')
    p.add_argument('--model',   default='m', choices=['n', 's', 'm', 'l', 'x'],
                   help='YOLO11 model size (默认: m)')
    p.add_argument('--epochs',  type=int, default=150,
                   help='训练轮数 (默认: 150)')
    p.add_argument('--batch',   type=int, default=8,
                   help='批次大小 (默认: 8)')
    p.add_argument('--imgsz',   type=int, default=800,
                   help='图像大小 (默认: 800)')
    p.add_argument('--device',  default='0',
                   help='CUDA device id 或 "cpu" (默认: 0)')
    p.add_argument('--name',    default='train',
                   help='实验名称 (默认: train)')
    p.add_argument('--workers', type=int, default=8,
                   help='Dataloader workers (默认: 8)')
    p.add_argument('--resume',  action='store_true',
                   help='从上次断点继续训练')
    return p.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    script_dir  = Path(__file__).parent
    data_yaml   = script_dir / 'det_data.yaml'
    weights_dir = script_dir.parent / 'weights'
    pretrained  = weights_dir / f'yolo11{args.model}.pt'

    print('=' * 70)
    print('🚀  AP 正面模型训练 — 解剖标志点检测 (Detect)')
    print('    CR / CL / IR / IL / SR / SL  →  6 类 YOLO 检测')
    print('=' * 70)
    print(f'  模型大小   : YOLO11{args.model}')
    print(f'  预训练权重 : {pretrained}')
    print(f'  训练轮数   : {args.epochs}')
    print(f'  批次大小   : {args.batch}')
    print(f'  图像大小   : {args.imgsz}')
    print(f'  设备       : {args.device}')
    print(f'  实验名称   : {args.name}')
    print(f'  数据集     : {data_yaml}')
    print('=' * 70)
    print()

    if not data_yaml.exists():
        raise FileNotFoundError(
            f'数据集配置文件不存在: {data_yaml}\n'
            f'请先运行: python convert_pose_to_detection.py'
        )

    # ── 加载模型 ──────────────────────────────────────────────────
    if args.resume:
        ckpt = script_dir / f'runs/detect/{args.name}/weights/last.pt'
        if ckpt.exists():
            print(f'从检查点继续: {ckpt}')
            model = YOLO(str(ckpt))
        else:
            print('⚠️  未找到检查点，从头开始')
            model = YOLO(str(pretrained) if pretrained.exists()
                         else f'yolo11{args.model}.pt')
    else:
        if pretrained.exists():
            print(f'加载预训练权重: {pretrained}')
            model = YOLO(str(pretrained))
        else:
            print('⚠️  未找到本地权重，尝试从 ultralytics 下载')
            model = YOLO(f'yolo11{args.model}.pt')

    # ── 开始训练 ──────────────────────────────────────────────────
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,

        # 优化器 (AdamW 对医学小目标收敛更稳)
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        # multi_scale=True 会随机缩小图像，导致特征图 1×1 时 BatchNorm 崩溃
        multi_scale=False,
        rect=False,

        # 数据增强 (正面脊柱: 不上下翻转)
        flipud=0.0,
        fliplr=0.5,
        degrees=3.0,          # 轻微旋转，X 光片一般较正
        translate=0.1,
        scale=0.5,
        perspective=0.0001,

        # 颜色增强 (X 光片保守值)
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,

        # 混合增强
        mosaic=1.0,
        close_mosaic=30,
        mixup=0.0,            # 解剖点位置敏感，不做 mixup
        copy_paste=0.0,
        erasing=0.3,
        auto_augment='randaugment',

        # 早停 & 保存
        patience=50,
        save=True,
        save_period=10,
        amp=False,

        # 输出
        project=str(script_dir / 'runs/detect'),
        name=args.name,
        plots=True,
        verbose=True,
    )

    print()
    print('=' * 70)
    print('✅  训练完成！')
    print(f'   最佳模型: runs/detect/{args.name}/weights/best.pt')
    print('   推理提示: 取检测框 (x1+x2)/2, (y1+y2)/2 作为标志点坐标')
    print('=' * 70)


if __name__ == '__main__':
    main()
