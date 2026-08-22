#!/bin/bash
# AP 正面脊柱椎体角点检测模型训练脚本
# 任务: 18 个椎体类别, 每个 4 个角点关键点 (TL/TR/BR/BL)

set -e

# ── 默认参数 ──────────────────────────────────────────────────
MODEL="s"
EPOCHS=200
IMGSZ=640
BATCH=16
DEVICE="0"
NAME="train"
DATA="corner_data.yaml"
AUGMENTATION_PROFILE="standard"
RESUME_FLAG=""

show_help() {
cat << EOF
用法: $0 [选项]

选项:
  --model   <n|s|m|l>   模型大小 (默认: s)
  --epochs  <num>        训练轮数 (默认: 200)
  --imgsz   <size>       图像大小 (默认: 640)
  --batch   <size>       批次大小 (默认: 16)
  --device  <id>         GPU ID  (默认: 0)
  --name    <name>       实验名称 (默认: train)
  --data    <yaml>       数据配置 (默认: corner_data.yaml)
  --augmentation-profile <standard|roi_low>  增强预设
  --roi-mixed            使用原图+ROI混合数据和低增强预设
  --resume               从上次中断处继续
  --help                 显示帮助

预设配置 (覆盖以上参数):
  --quick      快速测试  : nano,   50  轮, 640px, batch=32
  --standard   标准训练  : small,  200 轮, 640px, batch=16
  --accurate   高精度    : medium, 300 轮, 800px, batch=8
  --best       最佳性能  : large,  400 轮, 1024px, batch=4

示例:
  $0 --standard
  $0 --model m --epochs 300 --device 0
  $0 --best --device 0,1
  $0 --roi-mixed --imgsz 800 --name corner_roi_mixed_v1
EOF
}

# ── 解析参数 ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)   MODEL="$2";  shift 2 ;;
    --epochs)  EPOCHS="$2"; shift 2 ;;
    --imgsz)   IMGSZ="$2";  shift 2 ;;
    --batch)   BATCH="$2";  shift 2 ;;
    --device)  DEVICE="$2"; shift 2 ;;
    --name)    NAME="$2";   shift 2 ;;
    --data)    DATA="$2"; shift 2 ;;
    --augmentation-profile) AUGMENTATION_PROFILE="$2"; shift 2 ;;
    --roi-mixed)
      DATA="corner_data_roi_mixed.yaml"; AUGMENTATION_PROFILE="roi_low"; NAME="corner_roi_mixed_v1"; shift ;;
    --resume)  RESUME_FLAG="--resume"; shift ;;
    --quick)
      MODEL="n"; EPOCHS=50; IMGSZ=640; BATCH=32; NAME="quick_test"; shift ;;
    --standard)
      MODEL="s"; EPOCHS=200; IMGSZ=640; BATCH=16; NAME="standard"; shift ;;
    --accurate)
      MODEL="m"; EPOCHS=200; IMGSZ=800; BATCH=8; NAME="high_accuracy"; shift ;;
    --best)
      MODEL="l"; EPOCHS=200; IMGSZ=800; BATCH=4; NAME="best_performance"; shift ;;
    --help) show_help; exit 0 ;;
    *) echo "❌ 未知参数: $1"; show_help; exit 1 ;;
  esac
done

# ── 环境检查 ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "🚀  AP 正面模型训练 — 椎体角点检测 (Corner)"
echo "=========================================================="
echo "  模型大小 : YOLO11${MODEL}-pose"
echo "  训练轮数 : ${EPOCHS}"
echo "  图像大小 : ${IMGSZ}"
echo "  批次大小 : ${BATCH}"
echo "  GPU 设备 : ${DEVICE}"
echo "  实验名称 : ${NAME}"
echo "  数据配置 : ${DATA}"
echo "  增强预设 : ${AUGMENTATION_PROFILE}"
echo "  继续训练 : ${RESUME_FLAG:-否}"
echo "=========================================================="
echo ""

if [ ! -f "$DATA" ]; then
  echo "❌ 错误: $DATA 不存在"; exit 1
fi

if ! python3 -c "import ultralytics" 2>/dev/null; then
  echo "❌ 错误: 未安装 ultralytics — pip install ultralytics"; exit 1
fi

# ── 开始训练 ──────────────────────────────────────────────────
python3 train_corner.py \
  --model  "$MODEL"  \
  --epochs "$EPOCHS" \
  --imgsz  "$IMGSZ"  \
  --batch  "$BATCH"  \
  --device "$DEVICE" \
  --name   "$NAME"   \
  --data   "$DATA"   \
  --augmentation-profile "$AUGMENTATION_PROFILE" \
  $RESUME_FLAG

echo ""
echo "=========================================================="
echo "✅  训练完成！"
echo "   最佳模型: runs/corner/${NAME}/weights/best.pt"
echo "   结果图表: runs/corner/${NAME}/results.png"
echo "=========================================================="
