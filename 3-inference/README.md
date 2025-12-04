# Spine Analysis API

脊柱X光片分析后端服务，基于 YOLO 模型检测躯干标志点和椎体角点，输出前端交互系统需要的 JSON 格式。

## 目录结构

```
3-inference/
├── app.py              # FastAPI 服务主文件
├── requirements.txt    # Python 依赖
├── README.md           # 本文件
└── weights/
    ├── pose.pt         # 躯干标志点检测模型 (6个关键点)
    └── pose_corner.pt  # 椎体角点检测模型 (18类椎体，每个4角点)
```

## 快速开始

### 1. 安装依赖

```bash
cd 3-inference
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 方式1: 直接运行
python app.py

# 方式2: 使用 uvicorn (支持热重载)
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问: http://localhost:8000

### 3. API 文档

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API 接口

### 健康检查

```bash
GET /health
```

**响应:**
```json
{
  "status": "ok",
  "pose_model": true,
  "pose_corner_model": true
}
```

### 图片推理

```bash
POST /predict
```

**请求:**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@spine_xray.jpg" \
     -F "image_id=IMG018"
```

**响应 (前端交互系统格式):**
```json
{
  "imageId": "IMG018",
  "measurements": [
    {
      "type": "T1 Tilt",
      "points": [{"x": 1.78, "y": 170.79}, {"x": 61.54, "y": 202.90}]
    },
    {
      "type": "Cobb",
      "points": [
        {"x": -3.57, "y": 98.55}, {"x": 73.13, "y": 105.69},
        {"x": 0.89, "y": 72.69}, {"x": 71.35, "y": 54.85}
      ]
    },
    {
      "type": "RSH",
      "points": [{"x": 8.03, "y": 28.09}, {"x": 76.70, "y": 9.36}]
    },
    {
      "type": "Pelvic",
      "points": [{"x": 6.24, "y": -85.17}, {"x": -52.62, "y": -61.09}]
    },
    {
      "type": "Sacral",
      "points": [{"x": 16.05, "y": -142.25}, {"x": 9.81, "y": -111.04}]
    },
    {
      "type": "AVT",
      "points": [{"x": -18.73, "y": -177.04}, {"x": 12.91, "y": -177.04}]
    },
    {
      "type": "TS",
      "points": [{"x": -106.13, "y": 107.47}, {"x": 12.91, "y": 107.47}]
    }
  ]
}
```

## 指标说明

| type | 中文名 | 点位说明 |
|------|--------|----------|
| `T1 Tilt` | T1倾斜角 | T1(V1)上终板左右端点 |
| `Cobb` | Cobb角 | 上端椎上终板 + 下端椎下终板 (4点) |
| `RSH` | 两肩倾斜角 | 左右锁骨最高点 (CR, CL) |
| `Pelvic` | 骨盆倾斜角 | 左右髂骨最高点 (IR, IL) |
| `Sacral` | 骶骨倾斜角 | 骶一上终板左右缘点 (SR, SL) |
| `AVT` | 顶椎偏移 | 顶椎中心 → CSVL |
| `TS` | 躯干偏移 | C7中心 → CSVL |

## 模型信息

| 模型 | 文件 | 输出 |
|------|------|------|
| Pose | `weights/pose.pt` | 6个躯干关键点: CR, CL, IR, IL, SR, SL |
| Pose Corner | `weights/pose_corner.pt` | 18类椎体(V0-V17)，每个4角点: TL, TR, BR, BL |

## Docker 部署 (可选)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t spine-api .
docker run -p 8000:8000 spine-api
```

