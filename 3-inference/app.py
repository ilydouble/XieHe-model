#!/usr/bin/env python3
"""
Spine Analysis API Server

提供脊柱 X 光片分析的 REST API 服务，包含：
- Pose: 躯干标志点检测 (6个关键点)
- Pose Corner: 椎骨角点检测 (每个椎骨4个角点)

启动服务:
    python app.py

    # 或使用 uvicorn (支持热重载)
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

API 接口:
    POST /predict
    - 上传图片，返回前端交互系统需要的 annotations 格式 JSON

    GET /health
    - 健康检查
"""

import sys
import math
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import cv2
from ultralytics import YOLO

# ==================== 配置 ====================
POSE_MODEL_PATH = "weights/pose.pt"
POSE_CORNER_MODEL_PATH = "weights/pose_corner.pt"
CONF_THRESHOLD = 0.5

# ==================== 数据模型 ====================
class PointXY(BaseModel):
    x: float
    y: float

class Measurement(BaseModel):
    type: str
    points: List[PointXY]

class AnnotationsResponse(BaseModel):
    """前端交互系统需要的格式"""
    imageId: str
    measurements: List[Measurement]

# ==================== 初始化 ====================
app = FastAPI(
    title="Spine Analysis API",
    description="脊柱 X 光片分析服务：躯干标志点 + 椎骨角点检测",
    version="1.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型变量
pose_model = None
pose_corner_model = None

# ==================== 模型加载 ====================
def load_models():
    """加载模型（延迟加载）"""
    global pose_model, pose_corner_model
    
    if pose_model is None and Path(POSE_MODEL_PATH).exists():
        print(f"Loading Pose model: {POSE_MODEL_PATH}")
        pose_model = YOLO(POSE_MODEL_PATH)
    
    if pose_corner_model is None and Path(POSE_CORNER_MODEL_PATH).exists():
        print(f"Loading Pose Corner model: {POSE_CORNER_MODEL_PATH}")
        pose_corner_model = YOLO(POSE_CORNER_MODEL_PATH)

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_models()
    print("✅ Server started!")

# ==================== 推理函数 ====================
def infer_pose(img: np.ndarray) -> Dict[str, dict]:
    """
    Pose 模型推理 - 6个躯干标志点
    返回: {keypoint_name: {x, y, confidence}}
    """
    if pose_model is None:
        return {}

    results = pose_model.predict(img, verbose=False)
    result = results[0]

    keypoint_names = ['CR', 'CL', 'IR', 'IL', 'SR', 'SL']
    pose_data = {}

    if result.keypoints is not None and len(result.keypoints) > 0:
        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
        box_confs = result.boxes.conf.cpu().numpy()

        # 取置信度最高的检测结果
        best_idx = 0
        best_conf = 0
        for obj_idx in range(len(keypoints)):
            if box_confs[obj_idx] > best_conf and box_confs[obj_idx] >= CONF_THRESHOLD:
                best_conf = box_confs[obj_idx]
                best_idx = obj_idx

        if best_conf >= CONF_THRESHOLD:
            for kpt_idx in range(min(6, len(keypoints[best_idx]))):
                x, y = keypoints[best_idx][kpt_idx]
                conf = confidences[best_idx][kpt_idx] if confidences is not None else 1.0
                pose_data[keypoint_names[kpt_idx]] = {
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(conf)
                }

    return pose_data

def infer_pose_corner(img: np.ndarray) -> Dict[str, dict]:
    """
    Pose Corner 模型推理 - 椎骨4角点
    返回: {class_name: {corners: {top_left, top_right, bottom_right, bottom_left}}}
    """
    if pose_corner_model is None:
        return {}

    results = pose_corner_model.predict(img, verbose=False)
    result = results[0]

    class_names = pose_corner_model.names
    corner_keys = ["top_left", "top_right", "bottom_right", "bottom_left"]
    vertebrae = {}

    if result.keypoints is not None and len(result.keypoints) > 0:
        keypoints = result.keypoints.data.cpu().numpy()
        boxes = result.boxes

        for i, kpts in enumerate(keypoints):
            conf = float(boxes.conf[i])
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(boxes.cls[i])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"V{cls_id}"

            corners = {}
            for j, (x, y, v) in enumerate(kpts):
                corners[corner_keys[j]] = {"x": float(x), "y": float(y), "conf": float(v)}

            # 计算中点和中心
            tl, tr = corners["top_left"], corners["top_right"]
            bl, br = corners["bottom_left"], corners["bottom_right"]

            corners["top_mid"] = {"x": (tl["x"] + tr["x"]) / 2, "y": (tl["y"] + tr["y"]) / 2}
            corners["bottom_mid"] = {"x": (bl["x"] + br["x"]) / 2, "y": (bl["y"] + br["y"]) / 2}
            corners["center"] = {
                "x": (tl["x"] + tr["x"] + bl["x"] + br["x"]) / 4,
                "y": (tl["y"] + tr["y"] + bl["y"] + br["y"]) / 4
            }

            vertebrae[cls_name] = {"corners": corners, "confidence": conf, "class_id": cls_id}

    return vertebrae


# ==================== 转换为前端格式 ====================
def calc_angle(p1: dict, p2: dict) -> float:
    """计算两点连线与水平线的夹角（度）"""
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    return math.degrees(math.atan2(dy, dx))


def convert_to_annotations(
    pose_data: Dict[str, dict],
    vertebrae_data: Dict[str, dict],
    image_id: str
) -> dict:
    """
    将模型输出转换为前端交互系统需要的 annotations 格式
    """
    measurements = []

    # 1. T1 Tilt - T1(V1)上终板左右端点
    if "V1" in vertebrae_data:
        v1 = vertebrae_data["V1"]["corners"]
        measurements.append({
            "type": "T1 Tilt",
            "points": [
                {"x": v1["top_left"]["x"], "y": v1["top_left"]["y"]},
                {"x": v1["top_right"]["x"], "y": v1["top_right"]["y"]}
            ]
        })

    # 2. Cobb - 找上端椎和下端椎
    if vertebrae_data:
        max_tilt = float('-inf')
        min_tilt = float('inf')
        upper_v = None
        lower_v = None

        for name, data in vertebrae_data.items():
            corners = data["corners"]
            tilt = calc_angle(corners["top_left"], corners["top_right"])
            if tilt > max_tilt:
                max_tilt = tilt
                upper_v = corners
            if tilt < min_tilt:
                min_tilt = tilt
                lower_v = corners

        if upper_v and lower_v:
            measurements.append({
                "type": "Cobb",
                "points": [
                    {"x": upper_v["top_left"]["x"], "y": upper_v["top_left"]["y"]},
                    {"x": upper_v["top_right"]["x"], "y": upper_v["top_right"]["y"]},
                    {"x": lower_v["bottom_left"]["x"], "y": lower_v["bottom_left"]["y"]},
                    {"x": lower_v["bottom_right"]["x"], "y": lower_v["bottom_right"]["y"]}
                ]
            })

    # 3. RSH (两肩倾斜角) - CR, CL
    if "CR" in pose_data and "CL" in pose_data:
        measurements.append({
            "type": "RSH",
            "points": [
                {"x": pose_data["CR"]["x"], "y": pose_data["CR"]["y"]},
                {"x": pose_data["CL"]["x"], "y": pose_data["CL"]["y"]}
            ]
        })

    # 4. Pelvic (骨盆倾斜角) - IR, IL
    if "IR" in pose_data and "IL" in pose_data:
        measurements.append({
            "type": "Pelvic",
            "points": [
                {"x": pose_data["IR"]["x"], "y": pose_data["IR"]["y"]},
                {"x": pose_data["IL"]["x"], "y": pose_data["IL"]["y"]}
            ]
        })

    # 5. Sacral (骶骨倾斜角) - SR, SL
    if "SR" in pose_data and "SL" in pose_data:
        measurements.append({
            "type": "Sacral",
            "points": [
                {"x": pose_data["SR"]["x"], "y": pose_data["SR"]["y"]},
                {"x": pose_data["SL"]["x"], "y": pose_data["SL"]["y"]}
            ]
        })

    # 计算 CSVL (骶一中点的x坐标)
    csvl_x = None
    if "SR" in pose_data and "SL" in pose_data:
        csvl_x = (pose_data["SR"]["x"] + pose_data["SL"]["x"]) / 2

    # 6. AVT (顶椎偏移) - 顶椎中心 和 CSVL上对应点
    if vertebrae_data and csvl_x is not None:
        max_offset = 0
        apex_center = None
        for name, data in vertebrae_data.items():
            center = data["corners"]["center"]
            offset = abs(center["x"] - csvl_x)
            if offset > max_offset:
                max_offset = offset
                apex_center = center

        if apex_center:
            measurements.append({
                "type": "AVT",
                "points": [
                    {"x": apex_center["x"], "y": apex_center["y"]},
                    {"x": csvl_x, "y": apex_center["y"]}
                ]
            })

    # 7. TS (躯干偏移/C7偏移) - C7(V0)中心 和 CSVL上对应点
    if "V0" in vertebrae_data and csvl_x is not None:
        c7_center = vertebrae_data["V0"]["corners"]["center"]
        measurements.append({
            "type": "TS",
            "points": [
                {"x": c7_center["x"], "y": c7_center["y"]},
                {"x": csvl_x, "y": c7_center["y"]}
            ]
        })

    return {
        "imageId": image_id,
        "measurements": measurements
    }


# ==================== API 路由 ====================
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "pose_model": pose_model is not None,
        "pose_corner_model": pose_corner_model is not None
    }


@app.post("/predict", response_model=AnnotationsResponse)
async def predict(
    file: UploadFile = File(...),
    image_id: Optional[str] = Query(None, description="图片ID，默认使用文件名")
):
    """
    上传图片进行推理

    返回前端交互系统需要的 annotations 格式 JSON，包含:
    - T1 Tilt: T1倾斜角的两个端点
    - Cobb: 上端椎和下端椎的终板端点（4个点）
    - RSH: 两肩倾斜角的两个点
    - Pelvic: 骨盆倾斜角的两个点
    - Sacral: 骶骨倾斜角的两个点
    - AVT: 顶椎偏移的两个点
    - TS: 躯干偏移的两个点
    """
    # 检查文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 读取图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    # 生成 image_id
    if image_id is None:
        # 从文件名生成，去掉扩展名
        image_id = Path(file.filename).stem if file.filename else "IMG001"

    # 推理
    pose_data = infer_pose(img)
    vertebrae_data = infer_pose_corner(img)

    # 转换为前端格式
    result = convert_to_annotations(pose_data, vertebrae_data, image_id)

    return result


# ==================== 主入口 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

