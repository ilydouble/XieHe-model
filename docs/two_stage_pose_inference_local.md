# 六点 Pose 本地两阶段推理

本入口只用于本地验证，不修改 `xiehe-system` 线上推理。流程是：原图首轮 Pose 推理，按最高置信目标框扩展 ROI，在 ROI 上二次推理，再把 bbox 和六点坐标映射回原图。首轮低置信、ROI 无效或二次漏检时会自动退回首轮结果，并在报告中记录原因。

## 新模型训练完成后的 175 张评估

假设训练命令使用名称 `roi_mixed_v1`：

```bash
cd /Users/liruirui/Documents/code/spine/Model
/opt/miniconda3/envs/cv/bin/python 8-test_model/run_two_stage_pose.py \
  --image-dir datasets/pose_data/images/test \
  --label-dir datasets/pose_data/labels/test \
  --model 6-train_ap_model/runs/pose/roi_mixed_v1/weights/best.pt \
  --output-dir 8-test_model/two_stage_results/roi_mixed_v1 \
  --imgsz 800 \
  --roi-margin 0.20 \
  --roi-conf 0.25
```

输出目录必须为空或尚不存在。结果包括：

- `previews/`：每张图的首轮与最终结果左右对照图；
- `results.json`：逐图坐标、fallback 原因和汇总指标；
- `summary.csv`：便于筛选首轮/最终误差及肩点、下四点纵向偏差。

是否迁移到线上至少应同时满足：二阶段平均误差低于首轮；肩点和下四点的有符号 `dy` 更接近 0；175 张完整检出率不下降；fallback 比例可接受；最差样本人工复核无裁断或错误目标框。

## 当前旧权重冒烟结论

`best_performance-3` 是原图训练权重，不是 ROI 混合训练模型。3 张代表图已验证 ROI 裁剪、二次推理、原图坐标回写和可视化链路正常，但二阶段平均误差从 18.45 px 增至 28.46 px。因此该旧权重不得用于上线两阶段推理；应在新的 ROI 混合模型训练完成后执行上述 175 张评估。
