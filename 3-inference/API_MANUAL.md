# 脊柱X光片分析 API 操作手册

## 一、环境准备

### 1.1 系统要求
- Python >= 3.8
- CUDA (可选，GPU加速)

### 1.2 安装依赖

```bash
cd 3-inference
pip install -r requirements.txt
```

如遇到缺少模块错误，手动安装：
```bash
pip install python-multipart uvicorn
```

## 二、启动服务

### 2.1 标准启动

```bash
cd 3-inference
python app.py
```

### 2.2 使用 uvicorn 启动（支持热重载）

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2.3 启动成功标志

```
Loading Pose model: weights/pose.pt
Loading Pose Corner model: weights/pose_corner.pt
✅ Server started!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 三、接口说明

### 3.1 健康检查

| 项目 | 内容 |
|------|------|
| URL | `GET /health` |
| 功能 | 检查服务和模型状态 |

**请求示例：**
```bash
curl http://localhost:8000/health
```

**响应示例：**
```json
{
  "status": "ok",
  "pose_model": true,
  "pose_corner_model": true
}
```

---

### 3.2 图片推理

| 项目 | 内容 |
|------|------|
| URL | `POST /predict` |
| 功能 | 上传X光片，返回测量点位 |
| Content-Type | `multipart/form-data` |

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | ✅ | X光片图片文件 |
| image_id | String | ❌ | 图片ID，默认使用文件名 |

**请求示例：**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@spine_xray.png" \
     -F "image_id=IMG001"
```

**响应格式：**
```json
{
  "imageId": "IMG001",
  "measurements": [
    {
      "type": "T1 Tilt",
      "points": [{"x": 245.32, "y": 312.45}, {"x": 298.76, "y": 318.92}]
    },
    ...
  ]
}
```

## 四、返回字段说明

### 4.1 measurements 数组

| type | 中文名 | 点数 | 说明 |
|------|--------|------|------|
| `T1 Tilt` | T1倾斜角 | 2 | V1（第二节椎体，通常为T1）上终板左右端点；顶部截断时回退到V0 |
| `Cobb` | Cobb角 | 4 | 上端椎上终板(2点) + 下端椎下终板(2点) |
| `RSH` | 两肩倾斜角 | 2 | 左右锁骨最高点 |
| `Pelvic` | 骨盆倾斜角 | 2 | 左右髂骨最高点 |
| `Sacral` | 骶骨倾斜角 | 2 | 骶一上终板左右缘点 |
| `AVT` | 顶椎偏移 | 2 | 顶椎中心 → CSVL上对应点 |
| `TS` | 躯干偏移 | 2 | V0（最顶部检测到的椎体，通常为C7）中心 → CSVL上对应点 |

### 4.3 椎体编号规则

推理时对 Pose Corner 模型输出做两步后处理，使编号更稳健：

1. **IoU 去重**：同一位置被不同类别重复预测时，保留置信度最高的结果
2. **按 y 坐标重新编号**：从图像顶部到底部依次编为 V0、V1、V2 ...，不依赖模型原始类别 ID

> 对于顶部/底部截断的图像，V0 为当前可见的最顶部椎体（不一定是 C7）。

### 4.2 坐标系说明
- 原点：图片左上角
- X轴：向右为正
- Y轴：向下为正
- 单位：像素

## 五、Python 调用示例

```python
import requests

# 上传图片进行推理
url = "http://localhost:8000/predict"
files = {"file": open("spine_xray.png", "rb")}
params = {"image_id": "IMG001"}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"图片ID: {result['imageId']}")
for m in result["measurements"]:
    print(f"{m['type']}: {m['points']}")
```

## 六、在线文档

服务启动后可访问自动生成的 API 文档：

| 文档类型 | 地址 |
|----------|------|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

## 七、常见问题

### Q1: 启动时报错 `No module named 'xxx'`
```bash
pip install python-multipart uvicorn
```

### Q2: 模型加载失败
检查 `weights/` 目录下是否存在模型文件：
```bash
ls -la weights/
# 应该有 pose.pt 和 pose_corner.pt
```

### Q3: 推理结果为空
- 确认上传的是有效的脊柱X光片
- 检查图片格式（支持 jpg, png）
- 尝试调整置信度阈值（修改 app.py 中的 `CONF_THRESHOLD`）

### Q4: T1 Tilt 或 TS 结果异常
- 若图像顶部截断（C7 不可见），V0 为最顶部可见椎体，T1 Tilt 会自动回退到 V0
- 检查图像是否包含足够的上段脊柱（至少包含 T1）

### Q4: 如何修改端口
```bash
uvicorn app:app --host 0.0.0.0 --port 9000
```

