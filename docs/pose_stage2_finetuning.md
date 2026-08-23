# 六点Pose二阶段专用微调

## 目标与边界

当前`best_performance-5`继续作为一阶段原图模型。二阶段不再复用同一权重，而是从该best权重初始化，在一阶段真实预测框生成的ROI-only数据上微调独立Pose精修模型。

- 不从零训练；加载`best_performance-5/weights/best.pt`继续优化。
- train使用一阶段预测ROI的确定性多视图；val使用与线上相同margin的单ROI。
- 不把原图混入二阶段数据，不生成二阶段test派生集。
- 模型和参数只在原始176张val完整两阶段链路上选择；175张test仅作最终一次评测。
- 本流程仍是六点Pose，不改成Detection，也不修改线上系统。

## 1. 生成一阶段预测ROI数据

先在少量样本上验证首轮权重、GPU和目录：

```bash
cd /root/autodl-tmp/yolo/Model
python scripts/build_pose_stage2_roi.py \
  --stage1-model 6-train_ap_model/runs/pose/best_performance-5/weights/best.pt \
  --device 0 \
  --limit 10
```

确认输出正常后正式生成：

```bash
python scripts/build_pose_stage2_roi.py \
  --stage1-model 6-train_ap_model/runs/pose/best_performance-5/weights/best.pt \
  --device 0 \
  --train-variants 2 \
  --margin 0.20 \
  --apply
```

正式输出为`datasets/pose_stage2_roi`：

- train预计最多`1404 × 2 = 2808`张ROI；
- val预计最多176张ROI；
- 无首轮检测或bbox低置信的源图会跳过并记录，不会伪造GT裁剪；
- 为保证YOLO标签仍可表示而扩大的预测框会记录为`expanded_for_truth`；
- `manifest.json`记录首轮权重、源文件、预测框、裁剪框及派生文件SHA-256。

生成后先检查manifest中的`skipped_source_count`和`truth_expanded_fraction`。如果扩框比例明显偏高，说明首轮预测ROI和标注范围仍不匹配，应先调大线上与数据生成共同使用的margin，而不是直接训练。

输出目录已存在时脚本会拒绝覆盖。需要改变配置时应保留旧manifest后使用一个新的`--output-root`，并同步新建对应YAML，避免不同ROI版本静默混合。

## 2. 检查微调配置

```bash
cd /root/autodl-tmp/yolo/Model/6-train_ap_model
./train_pose_stage2.sh --dry-run
```

默认配置：

- 初始化权重：`runs/pose/best_performance-5/weights/best.pt`
- 数据：`pose_data_stage2_roi.yaml`，只含预测ROI train/val
- epochs=100、batch=4、imgsz=800
- AdamW，`lr0=0.001`，余弦下降
- 默认`freeze=0`，即全模型低学习率微调
- 关闭mosaic、mixup、copy-paste、erasing和multi-scale
- 输出：`runs/pose_stage2/stage2_refiner_v1`

## 3. 正式微调

```bash
./train_pose_stage2.sh \
  --device 0 \
  --epochs 100 \
  --batch 4 \
  --imgsz 800 \
  --name stage2_refiner_v1
```

显存不足时只降低batch，不改变imgsz。发生中断后使用相同name恢复：

```bash
./train_pose_stage2.sh --name stage2_refiner_v1 --resume
```

训练产生的最佳权重为：

```text
6-train_ap_model/runs/pose_stage2/stage2_refiner_v1/weights/best.pt
```

## 4. 先在原始val上选择模型

```bash
cd /root/autodl-tmp/yolo/Model
python scripts/build_two_stage_pose_review.py \
  --image-dir datasets/pose_data/images/val \
  --label-dir datasets/pose_data/labels/val \
  --model 6-train_ap_model/runs/pose/best_performance-5/weights/best.pt \
  --second-model 6-train_ap_model/runs/pose_stage2/stage2_refiner_v1/weights/best.pt \
  --output-dir /root/autodl-tmp/pose_stage2_val_review \
  --imgsz 800 \
  --roi-margin 0.20 \
  --device 0
```

至少同时比较：平均/中位/P90误差、PCK@20、六个点各自误差、肩点和下四点有符号dy、跨度偏差、fallback和延迟。只有完整两阶段val优于首轮，才进入test。

## 5. 最终test评测

确定二阶段权重与margin后，在原始175张test上运行同一命令，仅把`images/val`、`labels/val`改为`images/test`、`labels/test`，并使用新的输出目录。评测manifest会分别记录首轮和二阶段权重及训练参数SHA-256。

如果二阶段在val或test仍没有稳定提升，线上继续使用一阶段模型；专用二阶段权重不应仅因已经训练完成就强制部署。
