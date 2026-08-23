# 六点Pose二阶段低成本微调

## 目标与边界

当前`best_performance-5`继续作为一阶段原图模型。二阶段从该权重初始化，直接复用仓库已有的1404张`pose_roi_views`及其标签，微调一个独立的六点Pose精修权重。

- 不从零训练，不重新生成、复制、硬链接或裁剪ROI图像。
- train只使用已有ROI；原始val只作训练过程健康监控。
- 默认30轮、AdamW、`lr0=0.0003`、全模型微调，每10轮保存一次检查点。
- 最终权重必须在原始176张val上通过完整“一阶段原图→ROI→二阶段→坐标回写”链路选择，不能只依赖训练日志中的自动`best.pt`。
- 原始175张test仅在参数和权重确定后作一次最终评测。
- 本流程仍是六点Pose，不改成Detection，也不修改线上系统。

该方案的代价是训练ROI来自GT安全框，而线上ROI来自一阶段预测框，二者存在一定域差异。它适合作为低成本验证；只有效果仍不足时，再运行本文末尾的预测ROI可选实验。

## 1. 检查配置，不生成数据

已有数据应为：

```text
datasets/pose_roi_views/images/train  1404张
datasets/pose_roi_views/labels/train  1404份
```

训练前先执行dry-run：

```bash
cd /root/autodl-tmp/yolo/Model/6-train_ap_model
./train_pose_stage2.sh --dry-run
```

默认配置：

- 初始化权重：`runs/pose/best_performance-5/weights/best.pt`
- 数据：`pose_data_stage2_existing_roi.yaml`
- epochs=30、batch=4、imgsz=800
- AdamW，`lr0=0.0003`，余弦下降
- `freeze=0`，即全模型低学习率微调
- `save_period=10`，保留周期检查点用于完整链路选择
- 关闭mosaic、mixup、copy-paste、erasing和multi-scale
- 输出：`runs/pose_stage2/stage2_existing_roi_v1`

## 2. 正式微调

```bash
./train_pose_stage2.sh --device 0
```

显存不足时只降低batch，不改变imgsz：

```bash
./train_pose_stage2.sh --device 0 --batch 2
```

发生中断后使用相同name恢复：

```bash
./train_pose_stage2.sh --name stage2_existing_roi_v1 --resume
```

检查点目录为：

```text
6-train_ap_model/runs/pose_stage2/stage2_existing_roi_v1/weights/
```

预计需要比较周期检查点、`best.pt`和`last.pt`。以实际存在的文件为准，不要因为某个文件名是`best.pt`就直接部署。

## 3. 在原始val上选择二阶段权重

对每个候选检查点运行同一套完整两阶段评测。例如评测`epoch10.pt`：

```bash
cd /root/autodl-tmp/yolo/Model
python scripts/build_two_stage_pose_review.py \
  --image-dir datasets/pose_data/images/val \
  --label-dir datasets/pose_data/labels/val \
  --model 6-train_ap_model/runs/pose/best_performance-5/weights/best.pt \
  --second-model 6-train_ap_model/runs/pose_stage2/stage2_existing_roi_v1/weights/epoch10.pt \
  --output-dir /root/autodl-tmp/pose_stage2_val_epoch10 \
  --imgsz 800 \
  --roi-margin 0.20 \
  --device 0
```

然后把`epoch10.pt`和输出目录依次换成`epoch20.pt`、`best.pt`、`last.pt`。至少比较：

- 平均、中位和P90像素误差、PCK@20；
- 六个点各自误差；
- 肩点与下方四点的有符号`dy`、上下跨度偏差；
- fallback比例；
- 一阶段、二阶段和总推理延迟。

只有完整两阶段val稳定优于一阶段基线，才进入test。若周期检查点优于自动`best.pt`，应选择周期检查点。

## 4. 最终test评测

权重和`roi-margin`在val上固定后，在原始175张test上运行同一命令，仅将`images/val`、`labels/val`改为`images/test`、`labels/test`，并使用新的输出目录。不得根据test结果继续调参或换权重。

如果二阶段仍没有稳定提升，线上继续使用一阶段模型；专用二阶段权重不应仅因已经训练完成就强制部署。

## 5. 可选：预测ROI对照实验

只有低成本方案证明ROI域差异仍是主要瓶颈时，才生成预测ROI数据：

```bash
cd /root/autodl-tmp/yolo/Model
python scripts/build_pose_stage2_roi.py \
  --stage1-model 6-train_ap_model/runs/pose/best_performance-5/weights/best.pt \
  --device 0 \
  --train-variants 2 \
  --margin 0.20 \
  --apply
```

生成后显式指定可选YAML，不改变低成本默认值：

```bash
cd 6-train_ap_model
./train_pose_stage2.sh \
  --data pose_data_stage2_roi.yaml \
  --name stage2_predicted_roi_v1 \
  --device 0
```

预测ROI输出目录已存在时生成脚本会拒绝覆盖。不同ROI配置应使用新的`--output-root`和对应YAML，避免数据版本静默混合。
