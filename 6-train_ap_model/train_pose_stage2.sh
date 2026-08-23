#!/bin/bash
# 六点Pose二阶段专用ROI微调：从首轮best.pt继续训练独立精修权重。

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

WEIGHTS="runs/pose/best_performance-5/weights/best.pt"
DATA="pose_data_stage2_existing_roi.yaml"
EPOCHS=30
BATCH=4
IMGSZ=800
DEVICE="0"
WORKERS=8
NAME="stage2_existing_roi_v1"
LR0=0.0003
FREEZE=0
SAVE_PERIOD=10
EXTRA_FLAGS=()

show_help() {
cat << EOF
用法: $0 [选项]

  --weights <best.pt>  一阶段最佳权重（默认: ${WEIGHTS}）
  --data <yaml>        二阶段ROI-only配置（默认: ${DATA}）
  --epochs <num>       微调轮数（默认: ${EPOCHS}）
  --batch <num>        batch（默认: ${BATCH}）
  --imgsz <num>        输入尺寸（默认: ${IMGSZ}）
  --device <id>        GPU设备（默认: ${DEVICE}）
  --workers <num>      数据加载进程（默认: ${WORKERS}）
  --name <name>        实验名称（默认: ${NAME}）
  --lr0 <float>        初始学习率（默认: ${LR0}）
  --freeze <num>       冻结前N层；0为全模型微调
  --save-period <num>  周期权重保存间隔（默认: ${SAVE_PERIOD}）
  --resume             从runs/pose_stage2/<name>/weights/last.pt继续
  --dry-run            只检查并打印配置，不启动训练
  --help               显示帮助

默认直接引用已有datasets/pose_roi_views，不生成或复制ROI图像。
如需以后运行预测ROI对照实验，可显式传--data pose_data_stage2_roi.yaml。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --lr0) LR0="$2"; shift 2 ;;
    --freeze) FREEZE="$2"; shift 2 ;;
    --save-period) SAVE_PERIOD="$2"; shift 2 ;;
    --resume) EXTRA_FLAGS+=("--resume"); shift ;;
    --dry-run) EXTRA_FLAGS+=("--dry-run"); shift ;;
    --help) show_help; exit 0 ;;
    *) echo "未知参数: $1"; show_help; exit 1 ;;
  esac
done

python3 train_pose_stage2.py \
  --weights "$WEIGHTS" \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --name "$NAME" \
  --lr0 "$LR0" \
  --freeze "$FREEZE" \
  --save-period "$SAVE_PERIOD" \
  "${EXTRA_FLAGS[@]}"
