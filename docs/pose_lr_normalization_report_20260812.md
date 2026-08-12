# 六点训练数据左右语义修正报告

日期：2026-08-12

## 统一规范

- 关键点通道顺序保持`CR、CL、IR、IL、SR、SL`。
- 画面左侧为`CL、IL、SL`，画面右侧为`CR、IR、SR`。
- 修正只交换`CR/CL、IR/IL、SR/SL`三对关键点三元组，不镜像图像，不修改bbox、坐标数值、可见性或数据划分。

## 执行结果

- 活动数据：`datasets/pose_data`，共685对图像和标签（train 533、val 77、test 75）。
- 修正前685份全部为旧约定`CR<CL、IR<IL、SR<SL`。
- 修正后685份全部为新规范`CL<CR、IL<IR、SL<SR`。
- 每份标签均通过逐槽位置换断言，bbox前5列保持不变，六个关键点的数值集合保持不变。
- 图像树SHA-256修正前后均为`8cadff8475cc22506321d6dcff5be9280704c3b0884bf065512808fa9c0f37b4`，证明图像未修改。
- 修正后Pose专项审计：685份合法六点Pose、0个格式/坐标/结构问题。

## 派生Detection同步

- `datasets/pose_det_data`中75份仍有当前Pose来源的标签已重新派生并同步。
- 同步后75份与Pose精确一致，转换不一致为0。
- 另外83份没有当前Pose来源的孤立旧标签未修改、未删除，也不应纳入后续训练或评估。

## 训练及后续导入

- `6-train_ap_model/pose_data.yaml`已增加`flip_idx: [1, 0, 3, 2, 5, 4]`，水平翻转增强时同步交换三对左右关键点。
- E盘正式新人工标注的主流方向已经符合新规范，后续导入应使用`--six-lr-policy block`，不再执行`swap_pairs`。
- 使用按新规范训练的模型替换XieHe-System旧模型时，必须同时取消运行时的六点标签交换，否则会发生二次反转。

## 恢复备份

修正前原标签位于`datasets/pose_data_lr_backup_20260812/`：

- `labels/`：685份Pose原标签；
- `detection_labels/`：75份Detection原标签；
- `manifest.json`和`detection_manifest.json`：逐文件修正前后SHA-256及整体摘要；
- `README.md`：恢复说明。

可重复执行工具为`scripts/normalize_pose_lr_labels.py`。工具默认仅预演，实际修改必须同时提供`--apply`与备份目录。
