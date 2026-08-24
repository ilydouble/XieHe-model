# Corner V18/V19增量恢复记录

仅恢复历史标签中的class 18（L6）和class 19（T13）；原有class 0–17、图像和split未改变。

受影响的train样本已同步刷新Corner ROI标签/裁剪，完整逐文件哈希见manifest.json。

## 验收结果

- 原图：2499份标签、45036个实例；L6=44、T13=11。
- ROI：1999份train标签、36022个实例；L6=34、T13=7。
- 54份原图与40份ROI的备份/修改后/图像哈希全部复验通过。
- Ultralytics 8.3.183按20类混合YAML实载3998/250/250张，0背景、0损坏、无cache落盘。
- 训练命令见`docs/corner_20class_retraining_20260824.md`。
