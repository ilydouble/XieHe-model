# Corner 20类重新训练说明（2026-08-24）

## 已完成的数据状态

- 原图数据仍为2499张，split保持1999/250/250，没有重新生成原图。
- 已从历史原标签增量恢复54张中的55行：V18/L6 44行、V19/T13 11行。
- Corner ROI只刷新受影响的40张train视图；其余1959张原样跳过，仅1张因原ROI裁到额外椎体而局部扩框。
- 混合训练YAML为20类，普通病例只含V0–V17是合法的；不需要复制或补空的V18/V19标签。
- 训练集实载为3998张（1999原图+1999 ROI），验证和测试仍各250张原图。

## 服务器训练命令

在仓库根目录执行：

```bash
cd 6-train_ap_model
bash train_corner.sh --roi-mixed --best --device 0 --name corner_20class_roi_mixed_v1
```

这会使用`corner_data_roi_mixed.yaml`、YOLO11l-pose、imgsz=800、batch=4、150轮和低干扰ROI增强配置。显存不足时把`--batch 4`改为`--batch 2`，或改用`--model m`。

这是类别数从18扩到20后的新训练，不要使用`--resume`续接旧18类run。脚本默认从通用YOLO11 Pose预训练权重开始，输出位于：

```text
6-train_ap_model/runs/corner/corner_20class_roi_mixed_v1/weights/best.pt
```

## 推理与评测约束

- V18=L6，解剖位置在L5下方。
- V19=T13，解剖位置在T12与L1之间；类别编号19不代表它是最下方椎体。
- 新模型必须保留预测的原始class ID。旧的“全部框按y排序后编号V0、V1……”只能作为普通18类病例的兼容诊断，不能作为20类主推理。
- 训练后先在当前250张test上分别报告基础V0–V17、L6、T13的召回和角点误差；L6/T13测试样本仅3/3张，必须同时给逐例结果，不能只看总体mAP。

## 可恢复记录

- 修改前备份：`datasets/corner_v18_v19_restore_backup_20260824/`
- 恢复manifest：`datasets/import_records/corner_v18_v19_restore_20260824/manifest.json`
- 历史追溯：`docs/corner_v18_v19_history_audit_20260824.md`
