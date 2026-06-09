#!/bin/bash
# AP 正面脊柱解剖标志点检测模型训练脚本
# 任务: 6 类目标检测 (CR / CL / IR / IL / SR / SL)
# 模型: YOLO11 Detect (非 Pose)

set -e

# ── 默认参数 ──────────────────────────────────────────────────
MODEL="m"
EPOCHS=150
IMGSZ=800
BATCH=8
DEVICE="0"
NAME="train"
RESUME_FLAG=""
BBOX_RATIO="0.04"

show_help() {
cat << EOF
用法: $0 [选项]

选项:
  --model       <n|s|m|l>   模型大小 (默认: m)
  --epochs      <num>        训练轮数 (默认: 150)
  --imgsz       <size>       图像大小 (默认: 800)
  --batch       <size>       批次大小 (默认: 8)
  --device      <id>         GPU ID   (默认: 0)
  --name        <name>       实验名称 (默认: train)
  --bbox-ratio  <ratio>      关键点 bbox 边长比例 (默认: 0.04 = 4%)
  --resume                   从上次中断处继续
  --convert                  训练前先运行数据转换脚本
  --help                     显示帮助

预设配置:
  --quick      快速测试  : nano,   30  轮, 640px, batch=16
  --standard   标准训练  : medium, 150 轮, 800px, batch=8
  --best       最佳性能  : large,  300 轮, 800px, batch=4

类别说明:
  class 0: CR  右侧锁骨最高点
  class 1: CL  左侧锁骨最高点
  class 2: IR  右侧髂骨最高点
  class 3: IL  左侧髂骨最高点
  class 4: SR  骶一上终板右缘点
  class 5: SL  骶一上终板左缘点

示例:
  $0 --convert --standard
  $0 --model m --epochs 150 --device 0
  $0 --best --device 0,1
EOF
}

# ── 解析参数 ──────────────────────────────────────────────────
CONVERT=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)      MODEL="$2";      shift 2 ;;
    --epochs)     EPOCHS="$2";     shift 2 ;;
    --imgsz)      IMGSZ="$2";      shift 2 ;;
    --batch)      BATCH="$2";      shift 2 ;;
    --device)     DEVICE="$2";     shift 2 ;;
    --name)       NAME="$2";       shift 2 ;;
    --bbox-ratio) BBOX_RATIO="$2"; shift 2 ;;
    --resume)     RESUME_FLAG="--resume"; shift ;;
    --convert)    CONVERT=1; shift ;;
    --quick)
      MODEL="n"; EPOCHS=30; IMGSZ=640; BATCH=16; NAME="quick_test"; shift ;;
    --standard)
      MODEL="m"; EPOCHS=150; IMGSZ=800; BATCH=8;  NAME="standard"; shift ;;
    --best)
      MODEL="l"; EPOCHS=300; IMGSZ=800; BATCH=4;  NAME="best_performance"; shift ;;
    --help) show_help; exit 0 ;;
    *) echo "❌ 未知参数: $1"; show_help; exit 1 ;;
  esac
done

# ── 环境检查 ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "🚀  AP 正面模型训练 — 解剖标志点检测 (Detect)"
echo "    CR / CL / IR / IL / SR / SL  →  6 类 YOLO 检测"
echo "=========================================================="
echo "  模型大小  : YOLO11${MODEL}"
echo "  训练轮数  : ${EPOCHS}"
echo "  图像大小  : ${IMGSZ}"
echo "  批次大小  : ${BATCH}"
echo "  GPU 设备  : ${DEVICE}"
echo "  实验名称  : ${NAME}"
echo "  bbox 比例 : ${BBOX_RATIO}"
echo "  继续训练  : ${RESUME_FLAG:-否}"
echo "=========================================================="
echo ""

if ! python3 -c "import ultralytics" 2>/dev/null; then
  echo "❌ 未安装 ultralytics — pip install ultralytics"; exit 1
fi

# ── 数据转换 (可选) ──────────────────────────────────────────
if [[ $CONVERT -eq 1 ]]; then
  echo "🔄  运行数据转换..."
  python3 convert_pose_to_detection.py --bbox-ratio "$BBOX_RATIO"
  echo ""
fi

if [ ! -f "det_data.yaml" ]; then
  echo "❌ det_data.yaml 不存在"; exit 1
fi

DET_DATA_PATH=$(python3 -c "
import yaml; d=yaml.safe_load(open('det_data.yaml')); print(d['path'])
" 2>/dev/null)

if [ ! -d "$DET_DATA_PATH" ]; then
  echo "❌ 数据集目录不存在: $DET_DATA_PATH"
  echo "   请先运行: python3 convert_pose_to_detection.py"
  echo "   或使用:   $0 --convert --standard"
  exit 1
fi

# ── 开始训练 ──────────────────────────────────────────────────
python3 train_detection.py \
  --model   "$MODEL"  \
  --epochs  "$EPOCHS" \
  --imgsz   "$IMGSZ"  \
  --batch   "$BATCH"  \
  --device  "$DEVICE" \
  --name    "$NAME"   \
  $RESUME_FLAG

echo ""
echo "=========================================================="
echo "✅  训练完成！"
echo "   最佳模型: runs/detect/${NAME}/weights/best.pt"
echo "   结果图表: runs/detect/${NAME}/results.png"
echo "   推理提示: 取检测框中心点作为解剖标志点坐标"
echo "=========================================================="
