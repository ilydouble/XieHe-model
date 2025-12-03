# Spine X-ray Analysis with YOLO

基于 YOLO 的脊柱 X 光片分析项目，包含椎骨分割和关键点检测任务。

## 项目结构

```
Model/
├── pose_data/              # 躯干标志点数据集 (6个关键点)
├── pose_corner_data/       # 椎骨角点数据集 (4个关键点)
├── seg_data/               # 椎骨分割数据集
├── pose-scripts/           # 关键点检测脚本
├── seg_scripts/            # 分割脚本
├── 1-data_report/          # 数据分析工具
├── 2-extract_corner/       # 角点提取工具
├── runs/                   # 训练结果
└── weights/                # 预训练权重
```

## 任务说明

| 任务 | 数据集 | 关键点数 | 说明 |
|------|--------|----------|------|
| Pose | `pose_data` | 6 | 躯干标志点 (锁骨、髂骨、骶骨) |
| Pose Corner | `pose_corner_data` | 4 | 椎骨四角点 (V0-V17, 18类) |
| Segmentation | `seg_data` | - | 椎骨实例分割 |

---

## Pose Corner (椎骨角点检测)

### 1. 训练

```bash
cd pose-scripts
python train_corner.py
```

主要参数可在脚本中修改：
- `epochs`: 训练轮数 (默认 300)
- `imgsz`: 图像尺寸 (默认 640)
- `batch`: 批量大小 (默认 16)

训练结果保存在 `runs/pose_corner/train/`

### 2. 可视化

```bash
cd pose-scripts

# 可视化单张图片
python visualize_corner.py --model ../runs/pose_corner/train/weights/best.pt \
                           --source test.jpg

# 可视化测试集
python visualize_corner.py --model ../runs/pose_corner/train/weights/best.pt \
                           --source ../pose_corner_data/images/test \
                           --output ./vis_results
```

参数说明：
- `--model`: 模型权重路径
- `--source`: 图片或目录路径
- `--output`: 可视化结果输出目录
- `--conf`: 置信度阈值 (默认 0.5)

### 3. 推理

```bash
cd pose-scripts

# 推理单张/多张图片
python inference_corner.py --source img1.jpg img2.jpg img3.jpg

# 推理测试集
python inference_corner.py --source ../pose_corner_data/images/test \
                           --output ../pose_corner_data/jsons
```

参数说明：
- `--model`: 模型权重路径 (默认 `../runs/pose_corner/train/weights/best.pt`)
- `--source`: 图片路径或目录 (支持多个)
- `--output`: JSON 输出目录 (默认 `../pose_corner_data/jsons`)
- `--conf`: 置信度阈值 (默认 0.5)

输出 JSON 格式：
```json
{
  "image": "xxx.jpg",
  "image_size": {"width": 1024, "height": 2048},
  "objects": [
    {
      "class_id": 5,
      "class_name": "V5",
      "confidence": 0.95,
      "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 280},
      "corners": [
        {"name": "top_left", "x": 105.5, "y": 205.2, "confidence": 0.98},
        {"name": "top_right", "x": 295.3, "y": 208.1, "confidence": 0.97},
        {"name": "bottom_right", "x": 292.8, "y": 275.6, "confidence": 0.96},
        {"name": "bottom_left", "x": 108.2, "y": 272.4, "confidence": 0.95}
      ]
    }
  ]
}
```

---

## Pose (躯干标志点检测)

### 训练

```bash
cd pose-scripts
python train.py
```

### 推理

```bash
cd pose-scripts
python infer_single_image.py --model ../runs/pose/train/weights/best.pt \
                             --source test.jpg
```

---

## Segmentation (椎骨分割)

### 训练

```bash
cd seg_scripts
python train.py
```

### 推理

```bash
cd seg_scripts
python infer.py --model ../runs/segment/train/weights/best.pt \
                --source test.jpg
```

---

## 数据分析

```bash
cd 1-data_report
python analyze_dataset.py
```

生成数据集统计报告和可视化图表。

---

## 模型导出 (可选)

```python
from ultralytics import YOLO

model = YOLO('runs/pose_corner/train/weights/best.pt')

# 导出为 ONNX
model.export(format='onnx', half=True)

# 导出为 TensorRT (需要 NVIDIA GPU)
model.export(format='engine', half=True)

# 导出为 INT8 量化 TensorRT
model.export(format='engine', int8=True, data='pose_corner_data/dataset.yaml')
```

