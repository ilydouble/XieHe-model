# 脊柱 X 光片模型算法介绍与实验结果

本文档整理当前仓库中 AP 正位全脊柱 X 光片分析模型的算法方案、数据预处理、训练参数、推理后处理和实验结果。信息来源包括训练脚本、数据配置、推理服务、数据分析报告和 `runs/` 下已保存的训练结果。

## 1. 任务与整体流程

本项目采用 Ultralytics YOLO11 系列模型完成脊柱 X 光片中的结构定位与测量点生成，核心任务分为三类：

| 模块 | 模型任务 | 输入 | 输出 | 主要用途 |
|---|---|---|---|---|
| Pose | YOLO11 Pose | AP 正位全脊柱 X 光片 | 1 个全脊柱目标 + 6 个躯干关键点 | 肩、骨盆、骶骨和 CSVL 参考线 |
| Pose Corner | YOLO11 Pose | AP 正位全脊柱 X 光片 | 多个椎体目标 + 每个椎体 4 个角点 | T1 Tilt、Cobb、AVT、TS 等测量 |
| Segmentation | YOLO11 Seg | AP 正位全脊柱 X 光片 | 每节椎体实例分割 mask | 早期分割任务，以及自动生成椎体角点数据 |

线上推理服务实际加载两套关键点模型：

- `3-inference/weights/pose.pt`：6 个躯干标志点。
- `3-inference/weights/pose_corner.pt`：椎体四角点。

推理流程为：

1. 读取上传图片，使用 OpenCV 解码为 BGR 图像。
2. Pose 模型检测 6 个躯干标志点：`CR`、`CL`、`IR`、`IL`、`SR`、`SL`。
3. Pose Corner 模型检测椎体 bbox 与四角点：`TL`、`TR`、`BR`、`BL`。
4. 对椎体检测结果进行 IoU 去重，并按 y 坐标从上到下重新编号为 `V0`、`V1`、`V2` 等。
5. 根据关键点计算前端需要的测量点，包括 `T1 Tilt`、`Cobb`、`RSH`、`Pelvic`、`Sacral`、`AVT`、`TS`。

## 2. Backbone 与模型结构

### 2.1 Pose 与 Pose Corner

Pose 和 Pose Corner 均使用 YOLO11 Pose 架构。已保存实验结果对应的训练配置为：

| 项目 | Pose | Pose Corner |
|---|---:|---:|
| 任务类型 | `pose` | `pose` |
| 模型配置 | `yolo11n-pose.yaml` | `yolo11n-pose.yaml` |
| 预训练权重 | `../weights/yolo11n-pose.pt` | `../weights/yolo11n-pose.pt` |
| 输入尺寸 | 640 | 640 |
| Epochs | 300 | 300 |
| Batch | 64 | 64 |
| Optimizer | SGD | SGD |
| Device | `0` | `0` |

YOLO11 Pose 的结构可概括为：

- Backbone：YOLO11 系列轻量卷积特征提取网络，实验权重使用 nano 规模。
- Neck：多尺度特征融合结构，用于兼顾全脊柱长距离结构和局部椎体边界。
- Head：检测框分支、类别分支、关键点分支联合预测。

两个 pose 模型的 head 差异主要来自数据配置：

| 模型 | 类别数 | 关键点形状 | 类别/点位定义 |
|---|---:|---:|---|
| Pose | 1 类 | `[6, 3]` | `spine`；CR、CL、IR、IL、SR、SL |
| Pose Corner | 18 类 | `[4, 3]` | `V0`-`V17`；TL、TR、BR、BL |

### 2.2 Segmentation

分割模型训练脚本使用：

- 模型：`yolo11n-seg.yaml`
- 预训练权重：`../weights/yolo11n-seg.pt`
- 数据：`../seg_data/dataset.yaml`
- 输入尺寸：640
- Epochs：300
- Batch：64
- Optimizer：SGD

分割模型不是当前 API 的主推理模型，但它用于椎体实例 mask 建模，也用于将分割标签转换为四角点关键点数据。

## 3. 数据集与标注定义

### 3.1 数据集规模

| 数据集 | 任务 | 图像数 | 目标数 | 类别数 | 划分 |
|---|---|---:|---:|---:|---|
| `pose_data` | 躯干关键点 | 1,565 | 1,565 | 1 | train 1,252 / val 157 / test 156 |
| `seg_data` | 椎体实例分割 | 1,387 | 25,021 | 20 | train 1,109 / val 139 / test 139 |
| `pose_corner_data` | 椎体角点 | 1,386 | 约 24,948 | 18 | 来源于 `seg_data` 自动提取 |

图像分辨率整体较高，`pose_data` 平均尺寸约为 2358 x 4486 px。新增 AP 数据报告中还记录了 743 张新增 AP 图像，平均尺寸约 3016 x 5377 px，主要覆盖 T1-L5，不含 C7。

### 3.2 Pose 关键点

| 索引 | 名称 | 定义 |
|---:|---|---|
| 0 | CR | 右侧锁骨最高点 |
| 1 | CL | 左侧锁骨最高点 |
| 2 | IR | 右侧髂骨最高点 |
| 3 | IL | 左侧髂骨最高点 |
| 4 | SR | 骶一上终板右缘点 |
| 5 | SL | 骶一上终板左缘点 |

Pose 数据集中关键点可见率为 56.17%，不可见率为 43.83%，缺失率为 0。

### 3.3 Pose Corner 关键点

每节椎体输出四个角点：

| 索引 | 名称 | 定义 |
|---:|---|---|
| 0 | TL | 左上角 / 上终板左端点 |
| 1 | TR | 右上角 / 上终板右端点 |
| 2 | BR | 右下角 / 下终板右端点 |
| 3 | BL | 左下角 / 下终板左端点 |

训练标签采用 YOLO Pose 格式：

```text
class x_center y_center width height kp1_x kp1_y kp1_v ... kp4_x kp4_y kp4_v
```

`flip_idx` 为 `[1, 0, 3, 2]`，用于水平翻转时交换 TL/TR、BR/BL。

### 3.4 从分割到角点的数据预处理

椎体角点数据由分割标签自动转换而来，主要步骤为：

1. 解析 YOLO segmentation polygon 或 NRRD segmentation mask。
2. 对每节椎体 mask / polygon 提取轮廓。
3. 使用 OpenCV `minAreaRect` 拟合最小外接旋转矩形。
4. 使用 `boxPoints` 得到四个顶点。
5. 按 TL、TR、BR、BL 顺序排序。
6. 坐标归一化到 `[0, 1]`。
7. 生成 YOLO Pose 标签，关键点 visibility 置为 `2`。

这种设计将椎体分割问题转为四角点回归问题，减少线上后处理对 mask 的依赖，并直接服务角度测量。

## 4. 训练参数

### 4.1 已保存实验结果参数

`runs/pose/train/args.yaml` 与 `runs/pose_corner/train/args.yaml` 显示，已有实验结果使用如下共同参数：

| 参数 | 值 |
|---|---:|
| `epochs` | 300 |
| `batch` | 64 |
| `imgsz` | 640 |
| `optimizer` | SGD |
| `lr0` | 0.01 |
| `lrf` | 0.01 |
| `momentum` | 0.937 |
| `weight_decay` | 0.0005 |
| `warmup_epochs` | 3.0 |
| `box` loss weight | 7.5 |
| `cls` loss weight | 0.5 |
| `dfl` loss weight | 1.5 |
| `pose` loss weight | 12.0 |
| `kobj` loss weight | 1.0 |
| `iou` | 0.7 |
| `max_det` | 300 |
| `amp` | true |
| `multi_scale` | false |

已保存实验中的增强参数：

| 增强 | 值 |
|---|---:|
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `degrees` | 0.0 |
| `translate` | 0.1 |
| `scale` | 0.5 |
| `perspective` | 0.0 |
| `flipud` | 0.0 |
| `fliplr` | 0.5 |
| `mosaic` | 1.0 |
| `close_mosaic` | 10 |
| `mixup` | 0.0 |
| `copy_paste` | 0.0 |
| `auto_augment` | randaugment |
| `erasing` | 0.4 |

### 4.2 当前训练脚本默认参数

`6-train_ap_model/train_pose.py` 与 `6-train_ap_model/train_corner.py` 是当前 AP 模型训练脚本，默认使用 `YOLO11s-pose`、200 epochs、batch 16、imgsz 640，并加入更强泛化策略：

- `multi_scale=True`：训练尺寸在 0.5x 到 1.5x 间随机缩放。
- `degrees=5.0`：小角度旋转。
- `perspective=0.0001`：轻微透视扰动。
- `mixup=0.1`、`copy_paste=0.1`：混合与拼贴增强。
- `dropout=0.1`：正则化。
- `weight_decay=0.001`：较已保存实验配置更强的权重衰减。
- `amp=False`：避免 AMP 自动检查触发额外权重下载。
- `close_mosaic=30`：延长 mosaic 使用阶段。

因此，实验结果表应对应 `runs/` 中已保存的 YOLO11n-pose 配置；当前训练脚本代表后续复训或新实验的推荐配置。

## 5. 推理后处理与测量逻辑

### 5.1 置信度阈值

API 中统一使用：

```text
CONF_THRESHOLD = 0.5
```

Pose 模型仅保留 bbox 置信度最高且超过阈值的全脊柱目标。Pose Corner 模型先过滤低置信度候选，再进行空间去重。

### 5.2 椎体去重与编号

Pose Corner 推理后不直接依赖模型原始类别编号，而是：

1. 对候选椎体按置信度降序排列。
2. 使用 IoU 阈值 0.3 做贪心去重，同一位置仅保留最高置信度结果。
3. 对保留椎体按四角点 y 坐标均值从上到下排序。
4. 重新编号为 `V0`、`V1`、`V2` 等。

这样可以降低类别错位、重复检测和顶部截断对测量结果的影响。对于顶部截断图像，`V0` 表示当前图像中最上方可见椎体，不一定是 C7。

### 5.3 派生点

对每个椎体四角点计算：

- `top_mid`：TL/TR 中点。
- `bottom_mid`：BL/BR 中点。
- `center`：四角点中心。

对躯干关键点计算：

- `S1_center`：SR/SL 中点。
- `CSVL`：经过 `S1_center.x` 的垂直参考线。

### 5.4 输出测量项

| 测量项 | 点位来源 | 逻辑 |
|---|---|---|
| T1 Tilt | `V1` 上终板 TL/TR；若缺失则用 `V0` | 第二节可见椎体通常对应 T1 |
| Cobb | 上端椎上终板 + 下端椎下终板 | 选择上终板倾角最大与最小的椎体 |
| RSH | CR/CL | 两侧锁骨最高点连线 |
| Pelvic | IR/IL | 两侧髂骨最高点连线 |
| Sacral | SR/SL | 骶一上终板两端点连线 |
| AVT | 顶椎中心到 CSVL | 选择相对 CSVL 横向偏移最大的椎体作为顶椎 |
| TS | `V0` 中心到 CSVL | 最顶部可见椎体中心到 CSVL |

## 6. 实验结果

### 6.1 Pose Corner 椎体角点模型

已保存结果路径：`runs/pose_corner/train/results.csv`。

| 指标 | 最佳关键点轮次 Epoch 131 | 最终 Epoch 300 |
|---|---:|---:|
| Box Precision | 0.9800 | 0.9787 |
| Box Recall | 0.9791 | 0.9720 |
| Box mAP50 | 0.9901 | 0.9878 |
| Box mAP50-95 | 0.7444 | 0.7469 |
| Pose Precision | 0.9802 | 0.9770 |
| Pose Recall | 0.9774 | 0.9703 |
| Pose mAP50 | 0.9907 | 0.9884 |
| Pose mAP50-95 | 0.9745 | 0.9702 |
| Train box loss | 0.9873 | 0.8290 |
| Train pose loss | 0.3078 | 0.1684 |
| Val box loss | 0.9480 | 0.9208 |
| Val pose loss | 0.1677 | 0.1511 |

结论：

- 椎体角点模型关键点 mAP50-95 最高达到 0.9745，最终仍保持 0.9702。
- Box mAP50 接近 0.99，说明椎体定位稳定。
- Box mAP50-95 约 0.75，明显低于关键点 mAP，符合椎体 bbox 边界在高 IoU 下更敏感的特点。

### 6.2 Pose 躯干六关键点模型

已保存结果路径：`runs/pose/train/results.csv`。

| 指标 | 最佳关键点轮次 Epoch 224 | 最终 Epoch 300 |
|---|---:|---:|
| Box Precision | 0.9996 | 0.9996 |
| Box Recall | 1.0000 | 1.0000 |
| Box mAP50 | 0.9950 | 0.9950 |
| Box mAP50-95 | 0.8817 | 0.8850 |
| Pose Precision | 0.9996 | 0.9996 |
| Pose Recall | 1.0000 | 1.0000 |
| Pose mAP50 | 0.9950 | 0.9950 |
| Pose mAP50-95 | 0.9860 | 0.9827 |
| Train box loss | 0.5504 | 0.3846 |
| Train pose loss | 0.2877 | 0.3254 |
| Val box loss | 0.5624 | 0.5573 |
| Val pose loss | 0.1342 | 0.1302 |

结论：

- 躯干六关键点模型关键点 mAP50-95 最高达到 0.9860，最终为 0.9827。
- 检测类指标接近饱和，说明单目标全脊柱框检测难度较低。
- 关键点可见率仅 56.17%，但模型在验证集上的关键点指标仍较高，说明可见点定位稳定。

### 6.3 Segmentation 模型

仓库包含 `seg_scripts/eval.py` 用于评估分割模型，可输出 Box 与 Mask 的 mAP50、mAP75、mAP50-95、mean Precision 和 mean Recall。但当前仓库未保存 `runs/seg/train/results.csv` 或对应评估输出，因此本文档不虚构分割实验数值。

当前可确认的分割训练设置为：

| 参数 | 值 |
|---|---:|
| 模型 | YOLO11n-seg |
| 预训练权重 | `../weights/yolo11n-seg.pt` |
| 数据 | `../seg_data/dataset.yaml` |
| Epochs | 300 |
| Batch | 64 |
| Image Size | 640 |
| Optimizer | SGD |

## 7. 适用性与注意事项

1. 当前线上推理依赖 Pose + Pose Corner，不依赖 Segmentation mask，因此响应更轻量。
2. Pose Corner 推理时按 y 轴重新编号，可以提升截断图像的鲁棒性，但也意味着 `V0` 是“最顶部可见椎体”，不是绝对解剖 C7。
3. 新增 AP 数据不含 C7，若加入复训，应关注 T1 Tilt 和 TS 对顶部椎体定义的影响。
4. 现有实验结果来自 YOLO11n-pose；当前训练脚本默认改为 YOLO11s-pose 且增强策略更强，复现实验时应明确使用哪套配置。
5. 分割数据中 L6、T13 属于少见变异类别，样本量较小；角点模型当前使用 V0-V17 的 18 类主序列。

## 8. 关键文件索引

| 内容 | 文件 |
|---|---|
| 数据集统计与 Pose Corner 原始总结 | `1-data_report/DATA_ANALYSIS_REPORT.md` |
| Pose 训练脚本 | `6-train_ap_model/train_pose.py` |
| Pose Corner 训练脚本 | `6-train_ap_model/train_corner.py` |
| Pose 数据配置 | `6-train_ap_model/pose_data.yaml` |
| Pose Corner 数据配置 | `6-train_ap_model/corner_data.yaml` |
| Segmentation 训练脚本 | `seg_scripts/train.py` |
| Segmentation 评估脚本 | `seg_scripts/eval.py` |
| 分割转角点工具 | `2-extract_corner/extract_corners.py` |
| 新增 NRRD 分割转角点工具 | `7-add_data_report/step2_seg_to_corners.py` |
| API 推理与测量后处理 | `3-inference/app.py` |
| Pose 实验结果 | `runs/pose/train/results.csv`、`runs/pose/train/args.yaml` |
| Pose Corner 实验结果 | `runs/pose_corner/train/results.csv`、`runs/pose_corner/train/args.yaml` |
