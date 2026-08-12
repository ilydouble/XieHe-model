# E盘新增正位数据训练集构建报告（2026-08-12）

## 结论

- 以`assignment_all.xlsx`中的815个唯一`Patient_short_id`作为正式人工标注白名单后，椎体角点任务有效新增为871份：`datasets/pose_corner_data`由743对增至1614对，分区为train 1465、val 74、test 75。
- 六点任务白名单内有810份候选；其中809份可按整批策略统一，1份混合关系样本需单独排除或复核。六点尚未写入`datasets/pose_data`。
- E盘规范化源目录保持不变；导入仅向目标train追加`eap_`前缀文件，现有val/test未改动。

## 输入与筛选

- 源目录：`/Volumes/E/spine_data/20260810-正面-已处理/training_export_normalized`
- 正式人工标注分配表：`/Volumes/E/spine_data/20260810-正面-已处理/assignment_all.xlsx`，815行正式分配、815个唯一患者ID。
- 任务质量清单：`/Volumes/E/spine_data/20260810-正面-已处理/模型独立训练清单_20260811`
- 当前源数据：1566张图像与1566份JSON，283个精确重复组，冗余图像284张。
- 旧任务清单曾选出六点812份和角点875份；增加assignment白名单后，六点排除2份测试图，角点排除4份测试图。
- 最终角点正式新增871份。六点最终为809份可统一候选、1份混合关系待复核。
- 其余不在assignment白名单、重复冲突、任务标注不完整或不合规样本均不导入。

## 导入前检查

944张唯一候选均通过以下检查：

- Pillow完整像素解码；
- 实际图像尺寸与JSON的`imageWidth/imageHeight`一致；
- 六点或72个椎体角点完整性、坐标范围和几何规则；
- YOLO Pose格式转换；
- 与目标数据集train/val/test全部图像的SHA-256排重；
- 目标文件名冲突检查。

结果：白名单内六点810份和角点871份均无样本级文件或格式错误，无目标精确重复，无目标文件名冲突。

## 已完成的角点导入

执行方式为脚本的`--tasks spine_pose --apply`模式。新增标签采用18行、class 0–17；每行按`TL/TR/BR/BL`输出4点，visibility为2。

| 分区 | 导入前 | 新增 | 导入后 |
|---|---:|---:|---:|
| train | 594 | 871 | 1465 |
| val | 74 | 0 | 74 |
| test | 75 | 0 | 75 |
| 合计 | 743 | 871 | 1614 |

最初导入的875份中发现4份不在assignment白名单：`1874_影像测试`及`64261572`的3份图像。对应图像与标签已按原始SHA-256移动到`datasets/quarantine_non_assignment_20260812/`，可恢复且未永久删除。

更正后复核结果：1614张图像全部可完整解码；图像/标签配对问题0；871份正式新增标签格式、类别、坐标、visibility和assignment归属问题0；全数据精确重复组0，跨split精确重复组0。

逐样本源文件、SHA-256、目标路径和状态记录在：

- `datasets/import_records/e_drive_20260812_corner/import_manifest.json`
- `datasets/import_records/e_drive_20260812_corner/report.md`
- `datasets/import_records/e_drive_20260812_assignment_filtered/import_manifest.json`
- `datasets/quarantine_non_assignment_20260812/manifest.json`

## 六点数据暂缓原因

现有`pose_data`的685份标签全部采用以下画面x方向关系：

- `CR < CL`
- `IR < IL`
- `SR < SL`

assignment白名单内810份六点候选中，809份的三对关系全部相反；另1份人工接受样本`2115_SCO2105P0022_20210514`为混合关系。因此新旧数据不能原样混合训练，脚本在`swap_pairs`或`mirror_image`策略下也会自动跳过该混合样本。

导入脚本提供三个显式策略：

1. `block`：默认策略，发现冲突时阻止实际写入；
2. `swap_pairs`：保持图像不变，交换CR/CL、IR/IL、SR/SL三对标签身份；
3. `mirror_image`：水平镜像图像，并对同名解剖点执行`x'=1-x`，不交换解剖身份。

建议先依据影像方向标记和标注规范确认CR/CL究竟表示患者解剖左右还是画面左右，再选择后两者之一。选择前，六点数据保持未导入状态。

## 可复现脚本

脚本：`scripts/import_e_drive_training_data.py`

默认执行预演，不修改数据；使用`--apply`才写入。传入`--assignment-xlsx assignment_all.xlsx`后，仅允许患者ID出现在正式分配表中的样本。脚本还支持任务单独选择、持久化哈希缓存、完整图像解码、目标全分区排重、临时文件原子替换、导入后SHA或镜像像素验证，以及幂等重跑。
