#!/usr/bin/env python3
"""Fine-tune a dedicated six-keypoint Pose refiner, reusing existing ROIs by default."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "runs/pose/best_performance-5/weights/best.pt"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="trained first-stage best.pt used to initialize refinement")
    parser.add_argument("--data", default="pose_data_stage2_existing_roi.yaml", help="ROI-only dataset YAML")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="stage2_existing_roi_v1")
    parser.add_argument("--lr0", type=float, default=0.0003, help="initial fine-tuning learning rate")
    parser.add_argument("--freeze", type=int, default=0, help="freeze first N layers; 0 fine-tunes the full model")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save-period", type=int, default=10, help="save periodic checkpoints for full-chain val selection")
    parser.add_argument("--resume", action="store_true", help="resume this stage-two run from last.pt")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the effective configuration without training")
    return parser.parse_args(argv)


def resolve_file(value: str | Path, script_dir: Path, description: str) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested] if requested.is_absolute() else [script_dir / requested, requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{description}不存在: {value}")


def refinement_augmentation() -> dict:
    return {
        "rect": False,
        "multi_scale": False,
        "degrees": 2.0,
        "translate": 0.03,
        "scale": 0.10,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.0,
        "hsv_s": 0.08,
        "hsv_v": 0.10,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "auto_augment": None,
        "dropout": 0.05,
    }


def build_train_config(args: argparse.Namespace, data_yaml: Path, project: Path) -> dict:
    config = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "amp": False,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "cos_lr": True,
        "patience": args.patience,
        "save_period": args.save_period,
        "seed": 0,
        "deterministic": True,
        "box": 7.5,
        "cls": 0.5,
        "pose": 12.0,
        "kobj": 2.0,
        "project": str(project),
        "name": args.name,
        "plots": True,
        "verbose": True,
        **refinement_augmentation(),
    }
    if args.freeze > 0:
        config["freeze"] = args.freeze
    if args.resume:
        config["resume"] = True
    return config


def effective_configuration(args: argparse.Namespace) -> tuple[Path, Path, Path, dict]:
    data_yaml = resolve_file(args.data, SCRIPT_DIR, "数据集配置文件")
    project = SCRIPT_DIR / "runs/pose_stage2"
    if args.resume:
        weights = project / args.name / "weights/last.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"二阶段断点不存在: {weights}")
    else:
        weights = resolve_file(args.weights, SCRIPT_DIR, "初始化权重")
    return weights, data_yaml, project, build_train_config(args, data_yaml, project)


def main(argv=None) -> None:
    args = parse_args(argv)
    weights, data_yaml, project, train_config = effective_configuration(args)
    printable = {
        "mode": "resume" if args.resume else "fine_tune",
        "initial_weights": str(weights),
        "data": str(data_yaml),
        "project": str(project),
        "train": train_config,
    }
    print("=" * 72)
    print("AP 六点Pose二阶段专用ROI微调")
    print(f"初始化权重 : {weights}")
    print(f"ROI数据    : {data_yaml}")
    print(f"输出实验   : {project / args.name}")
    print(f"学习率     : {args.lr0}")
    print(f"冻结层数   : {args.freeze}（0表示全模型微调）")
    print("=" * 72)
    if args.dry_run:
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return

    from ultralytics import YOLO

    model = YOLO(str(weights))
    previous_cwd = Path.cwd()
    os.chdir(SCRIPT_DIR)
    try:
        model.train(**train_config)
    finally:
        os.chdir(previous_cwd)
    print(f"训练完成，检查点目录：{project / args.name / 'weights'}")
    print("请用原始val完整两阶段链路比较周期权重、best.pt和last.pt，不要只依赖raw-val自动best。")


if __name__ == "__main__":
    main()
