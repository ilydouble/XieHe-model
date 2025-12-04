"""
脊柱X光片指标计算模块

交互系统点位定义 + 从模型输出计算点位 + 指标计算
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ============================================================
# 第一部分：点位定义（交互系统使用的标准点位）
# ============================================================

@dataclass
class Point:
    """二维点"""
    x: float
    y: float
    confidence: float = 1.0

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class VertebraPoints:
    """单个椎体的所有点位（交互系统标准定义）"""
    # 椎体标识
    class_id: int           # V0-V17
    class_name: str         # C7, T1, T2, ..., L5

    # 四个角点（模型直接输出）
    top_left: Point         # 上终板左端点 (TL)
    top_right: Point        # 上终板右端点 (TR)
    bottom_left: Point      # 下终板左端点 (BL)
    bottom_right: Point     # 下终板右端点 (BR)

    # 终板中点（由角点计算，存储供交互系统使用）
    top_mid: Point = None       # 上终板中点
    bottom_mid: Point = None    # 下终板中点

    # 椎体中心（由角点计算）
    center: Point = None

    def compute_midpoints(self):
        """从角点计算中点"""
        # 上终板中点
        self.top_mid = Point(
            x=(self.top_left.x + self.top_right.x) / 2,
            y=(self.top_left.y + self.top_right.y) / 2,
            confidence=min(self.top_left.confidence, self.top_right.confidence)
        )
        # 下终板中点
        self.bottom_mid = Point(
            x=(self.bottom_left.x + self.bottom_right.x) / 2,
            y=(self.bottom_left.y + self.bottom_right.y) / 2,
            confidence=min(self.bottom_left.confidence, self.bottom_right.confidence)
        )
        # 椎体中心
        self.center = Point(
            x=(self.top_left.x + self.top_right.x + self.bottom_left.x + self.bottom_right.x) / 4,
            y=(self.top_left.y + self.top_right.y + self.bottom_left.y + self.bottom_right.y) / 4,
            confidence=min(self.top_left.confidence, self.top_right.confidence,
                          self.bottom_left.confidence, self.bottom_right.confidence)
        )


@dataclass
class TrunkPoints:
    """躯干标志点（Pose模型输出）"""
    CR: Point   # 右侧锁骨最高点 (Clavicle Right)
    CL: Point   # 左侧锁骨最高点 (Clavicle Left)
    IR: Point   # 右侧髂骨最高点 (Iliac Right)
    IL: Point   # 左侧髂骨最高点 (Iliac Left)
    SR: Point   # 骶一上终板右缘点 (Sacrum Right)
    SL: Point   # 骶一上终板左缘点 (Sacrum Left)

    # 派生点（计算得到）
    S1_center: Point = None  # 骶一中点，用于CSVL

    def compute_derived_points(self):
        """计算派生点"""
        self.S1_center = Point(
            x=(self.SR.x + self.SL.x) / 2,
            y=(self.SR.y + self.SL.y) / 2,
            confidence=min(self.SR.confidence, self.SL.confidence)
        )


@dataclass
class SpinePoints:
    """完整的脊柱点位数据（交互系统使用）"""
    trunk: TrunkPoints                      # 躯干标志点
    vertebrae: Dict[str, VertebraPoints]    # 所有椎体点位 {V0: ..., V1: ..., ...}

    # 参考线
    csvl_x: float = None    # CSVL的x坐标（中央骶骨垂直线）

    def compute_all(self):
        """计算所有派生点"""
        # 计算躯干派生点
        self.trunk.compute_derived_points()
        self.csvl_x = self.trunk.S1_center.x

        # 计算每个椎体的中点
        for v in self.vertebrae.values():
            v.compute_midpoints()


# 椎体类别映射
VERTEBRA_NAMES = {
    0: "C7", 1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5",
    6: "T6", 7: "T7", 8: "T8", 9: "T9", 10: "T10", 11: "T11",
    12: "T12", 13: "L1", 14: "L2", 15: "L3", 16: "L4", 17: "L5"
}


# ============================================================
# 第二部分：从模型输出构建点位
# ============================================================

def build_spine_points(pose_result, pose_corner_result) -> SpinePoints:
    """
    从模型推理结果构建SpinePoints

    Args:
        pose_result: Pose模型输出，包含6个躯干关键点
        pose_corner_result: Pose Corner模型输出，包含各椎体的4个角点

    Returns:
        SpinePoints: 完整的脊柱点位数据
    """
    # 1. 解析躯干关键点
    # pose_result.keypoints: shape (1, 6, 3) -> [x, y, conf]
    kpts = pose_result[0].keypoints.data[0].cpu().numpy()  # (6, 3)

    trunk = TrunkPoints(
        CR=Point(x=kpts[0, 0], y=kpts[0, 1], confidence=kpts[0, 2]),
        CL=Point(x=kpts[1, 0], y=kpts[1, 1], confidence=kpts[1, 2]),
        IR=Point(x=kpts[2, 0], y=kpts[2, 1], confidence=kpts[2, 2]),
        IL=Point(x=kpts[3, 0], y=kpts[3, 1], confidence=kpts[3, 2]),
        SR=Point(x=kpts[4, 0], y=kpts[4, 1], confidence=kpts[4, 2]),
        SL=Point(x=kpts[5, 0], y=kpts[5, 1], confidence=kpts[5, 2]),
    )

    # 2. 解析椎体角点
    vertebrae = {}
    boxes = pose_corner_result[0].boxes.data.cpu().numpy()
    keypoints = pose_corner_result[0].keypoints.data.cpu().numpy()  # (N, 4, 3)

    for i, box in enumerate(boxes):
        class_id = int(box[5])
        class_name = VERTEBRA_NAMES.get(class_id, f"V{class_id}")
        kp = keypoints[i]  # (4, 3) -> [x, y, conf] for each corner

        vp = VertebraPoints(
            class_id=class_id,
            class_name=class_name,
            top_left=Point(x=kp[0, 0], y=kp[0, 1], confidence=kp[0, 2]),
            top_right=Point(x=kp[1, 0], y=kp[1, 1], confidence=kp[1, 2]),
            bottom_right=Point(x=kp[2, 0], y=kp[2, 1], confidence=kp[2, 2]),
            bottom_left=Point(x=kp[3, 0], y=kp[3, 1], confidence=kp[3, 2]),
        )
        vertebrae[f"V{class_id}"] = vp

    # 3. 构建完整的SpinePoints并计算派生点
    spine_points = SpinePoints(trunk=trunk, vertebrae=vertebrae)
    spine_points.compute_all()

    return spine_points


# ============================================================
# 第三部分：指标计算
# ============================================================

def calc_angle_from_points(p1: Point, p2: Point) -> float:
    """
    计算两点连线与水平线的夹角（度）
    正值表示p2比p1低（图像坐标系y向下）
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def calc_shoulder_tilt(spine: SpinePoints) -> dict:
    """
    计算两肩倾斜角
    使用点位: CR, CL
    """
    angle = calc_angle_from_points(spine.trunk.CR, spine.trunk.CL)
    return {
        "metric": "两肩倾斜角",
        "angle_deg": round(angle, 2),
        "points_used": ["CR", "CL"],
        "direction": "右肩高" if angle < 0 else "左肩高" if angle > 0 else "平衡"
    }


def calc_pelvic_obliquity(spine: SpinePoints) -> dict:
    """
    计算骨盆倾斜角
    使用点位: IR, IL
    """
    angle = calc_angle_from_points(spine.trunk.IR, spine.trunk.IL)
    return {
        "metric": "骨盆倾斜角",
        "angle_deg": round(angle, 2),
        "points_used": ["IR", "IL"],
        "direction": "右侧高" if angle < 0 else "左侧高" if angle > 0 else "平衡"
    }


def calc_sacral_obliquity(spine: SpinePoints) -> dict:
    """
    计算骶骨倾斜角
    使用点位: SR, SL
    """
    angle = calc_angle_from_points(spine.trunk.SR, spine.trunk.SL)
    return {
        "metric": "骶骨倾斜角",
        "angle_deg": round(angle, 2),
        "points_used": ["SR", "SL"],
        "direction": "右侧高" if angle < 0 else "左侧高" if angle > 0 else "平衡"
    }


def calc_T1_tilt(spine: SpinePoints) -> dict:
    """
    计算T1倾斜角
    使用点位: V1的上终板两端点 (top_left, top_right) 或 top_mid
    """
    if "V1" not in spine.vertebrae:
        return {"metric": "T1倾斜角", "error": "未检测到T1椎体"}

    v1 = spine.vertebrae["V1"]
    angle = calc_angle_from_points(v1.top_left, v1.top_right)
    return {
        "metric": "T1倾斜角",
        "angle_deg": round(angle, 2),
        "points_used": ["V1_top_left", "V1_top_right"],
        "top_mid": v1.top_mid.to_tuple()
    }


def calc_all_vertebra_tilts(spine: SpinePoints) -> Dict[str, float]:
    """
    计算所有椎体的上终板倾斜角
    用于Cobb角计算
    """
    tilts = {}
    for v_name, v in spine.vertebrae.items():
        angle = calc_angle_from_points(v.top_left, v.top_right)
        tilts[v_name] = round(angle, 2)
    return tilts


def calc_cobb_angle(spine: SpinePoints) -> dict:
    """
    计算Cobb角（主弯）
    使用点位: 所有椎体的上终板端点

    步骤:
    1. 计算每个椎体上终板倾斜角
    2. 找到最大正倾斜（上端椎）和最大负倾斜（下端椎）
    3. Cobb角 = |上端椎倾斜角 - 下端椎倾斜角|
    """
    tilts = calc_all_vertebra_tilts(spine)

    if not tilts:
        return {"metric": "Cobb角", "error": "未检测到椎体"}

    # 找最大和最小倾斜角
    max_item = max(tilts.items(), key=lambda x: x[1])
    min_item = min(tilts.items(), key=lambda x: x[1])

    cobb_angle = abs(max_item[1] - min_item[1])

    return {
        "metric": "Cobb角",
        "cobb_angle_deg": round(cobb_angle, 2),
        "upper_end_vertebra": max_item[0],
        "upper_end_tilt": max_item[1],
        "lower_end_vertebra": min_item[0],
        "lower_end_tilt": min_item[1],
        "all_tilts": tilts,
        "points_used": "所有椎体的top_left, top_right"
    }



def calc_C7_offset(spine: SpinePoints, pixel_to_mm: float = 1.0) -> dict:
    """
    计算C7偏移距离
    使用点位: V0的4角点(计算C7中心), SR+SL(计算CSVL)
    """
    if "V0" not in spine.vertebrae:
        return {"metric": "C7偏移距离", "error": "未检测到C7椎体"}

    c7 = spine.vertebrae["V0"]
    c7_center_x = c7.center.x
    csvl_x = spine.csvl_x

    offset_px = c7_center_x - csvl_x
    offset_mm = offset_px * pixel_to_mm

    return {
        "metric": "C7偏移距离",
        "offset_px": round(offset_px, 2),
        "offset_mm": round(offset_mm, 2),
        "direction": "右偏" if offset_px > 0 else "左偏" if offset_px < 0 else "居中",
        "C7_center": c7.center.to_tuple(),
        "CSVL_x": csvl_x
    }


def find_apex_vertebra(spine: SpinePoints) -> dict:
    """
    找到顶椎（偏离CSVL最远的椎体）
    使用点位: 所有椎体的center, CSVL
    """
    csvl_x = spine.csvl_x
    max_offset = 0
    apex = None

    for v_name, v in spine.vertebrae.items():
        offset = abs(v.center.x - csvl_x)
        if offset > max_offset:
            max_offset = offset
            apex = v_name
            apex_v = v

    if apex is None:
        return {"metric": "顶椎", "error": "未检测到椎体"}

    return {
        "metric": "顶椎",
        "apex_vertebra": apex,
        "apex_name": VERTEBRA_NAMES.get(int(apex[1:]), apex),
        "offset_px": round(max_offset, 2),
        "direction": "右偏" if apex_v.center.x > csvl_x else "左偏"
    }


# ============================================================
# 第四部分：导出点位（供交互系统使用）
# ============================================================

def export_points_to_dict(spine: SpinePoints) -> dict:
    """将SpinePoints导出为字典格式"""
    result = {
        "trunk_points": {
            "CR": {"x": spine.trunk.CR.x, "y": spine.trunk.CR.y},
            "CL": {"x": spine.trunk.CL.x, "y": spine.trunk.CL.y},
            "IR": {"x": spine.trunk.IR.x, "y": spine.trunk.IR.y},
            "IL": {"x": spine.trunk.IL.x, "y": spine.trunk.IL.y},
            "SR": {"x": spine.trunk.SR.x, "y": spine.trunk.SR.y},
            "SL": {"x": spine.trunk.SL.x, "y": spine.trunk.SL.y},
            "S1_center": {"x": spine.trunk.S1_center.x, "y": spine.trunk.S1_center.y},
        },
        "CSVL_x": spine.csvl_x,
        "vertebrae": {}
    }

    for v_name, v in spine.vertebrae.items():
        result["vertebrae"][v_name] = {
            "class_name": v.class_name,
            "top_left": {"x": v.top_left.x, "y": v.top_left.y},
            "top_right": {"x": v.top_right.x, "y": v.top_right.y},
            "bottom_left": {"x": v.bottom_left.x, "y": v.bottom_left.y},
            "bottom_right": {"x": v.bottom_right.x, "y": v.bottom_right.y},
            "top_mid": {"x": v.top_mid.x, "y": v.top_mid.y},
            "bottom_mid": {"x": v.bottom_mid.x, "y": v.bottom_mid.y},
            "center": {"x": v.center.x, "y": v.center.y},
        }

    return result


def calc_all_metrics(spine: SpinePoints, pixel_to_mm: float = 1.0) -> dict:
    """计算所有指标"""
    return {
        "shoulder_tilt": calc_shoulder_tilt(spine),
        "pelvic_obliquity": calc_pelvic_obliquity(spine),
        "sacral_obliquity": calc_sacral_obliquity(spine),
        "T1_tilt": calc_T1_tilt(spine),
        "cobb_angle": calc_cobb_angle(spine),
        "C7_offset": calc_C7_offset(spine, pixel_to_mm),
        "apex_vertebra": find_apex_vertebra(spine),
    }


# ============================================================
# 第五部分：转换为前端交互系统格式
# ============================================================

def convert_to_frontend_format(spine: SpinePoints, image_id: str = "IMG001") -> dict:
    """
    将SpinePoints转换为前端交互系统需要的格式

    前端格式:
    {
        "imageId": "IMG018",
        "measurements": [
            {"type": "T1 Tilt", "points": [{x, y}, {x, y}]},
            {"type": "Cobb", "points": [{x, y}, {x, y}, {x, y}, {x, y}]},
            {"type": "RSH", "points": [{x, y}, {x, y}]},
            {"type": "Pelvic", "points": [{x, y}, {x, y}]},
            {"type": "Sacral", "points": [{x, y}, {x, y}]},
            {"type": "AVT", "points": [{x, y}, {x, y}]},
            {"type": "TS", "points": [{x, y}, {x, y}]}
        ]
    }
    """
    measurements = []

    # 1. T1 Tilt - T1上终板左右端点
    if "V1" in spine.vertebrae:
        v1 = spine.vertebrae["V1"]
        measurements.append({
            "type": "T1 Tilt",
            "points": [
                {"x": v1.top_left.x, "y": v1.top_left.y},
                {"x": v1.top_right.x, "y": v1.top_right.y}
            ]
        })

    # 2. Cobb - 需要上端椎和下端椎的终板端点
    cobb_data = _find_cobb_vertebrae(spine)
    if cobb_data:
        upper_v, lower_v = cobb_data
        measurements.append({
            "type": "Cobb",
            "points": [
                # 上端椎上终板两端点
                {"x": upper_v.top_left.x, "y": upper_v.top_left.y},
                {"x": upper_v.top_right.x, "y": upper_v.top_right.y},
                # 下端椎下终板两端点
                {"x": lower_v.bottom_left.x, "y": lower_v.bottom_left.y},
                {"x": lower_v.bottom_right.x, "y": lower_v.bottom_right.y}
            ]
        })

    # 3. RSH (两肩倾斜角) - CR, CL
    measurements.append({
        "type": "RSH",
        "points": [
            {"x": spine.trunk.CR.x, "y": spine.trunk.CR.y},
            {"x": spine.trunk.CL.x, "y": spine.trunk.CL.y}
        ]
    })

    # 4. Pelvic (骨盆倾斜角) - IR, IL
    measurements.append({
        "type": "Pelvic",
        "points": [
            {"x": spine.trunk.IR.x, "y": spine.trunk.IR.y},
            {"x": spine.trunk.IL.x, "y": spine.trunk.IL.y}
        ]
    })

    # 5. Sacral (骶骨倾斜角) - SR, SL
    measurements.append({
        "type": "Sacral",
        "points": [
            {"x": spine.trunk.SR.x, "y": spine.trunk.SR.y},
            {"x": spine.trunk.SL.x, "y": spine.trunk.SL.y}
        ]
    })

    # 6. AVT (顶椎偏移) - 顶椎中心 和 CSVL上对应点
    apex_v = _find_apex_vertebra_obj(spine)
    if apex_v:
        measurements.append({
            "type": "AVT",
            "points": [
                {"x": apex_v.center.x, "y": apex_v.center.y},
                {"x": spine.csvl_x, "y": apex_v.center.y}  # CSVL上同高度的点
            ]
        })

    # 7. TS (躯干偏移/C7偏移) - C7中心 和 CSVL上对应点
    if "V0" in spine.vertebrae:
        c7 = spine.vertebrae["V0"]
        measurements.append({
            "type": "TS",
            "points": [
                {"x": c7.center.x, "y": c7.center.y},
                {"x": spine.csvl_x, "y": c7.center.y}  # CSVL上同高度的点
            ]
        })

    return {
        "imageId": image_id,
        "measurements": measurements
    }


def _find_cobb_vertebrae(spine: SpinePoints) -> Optional[Tuple[VertebraPoints, VertebraPoints]]:
    """
    找到Cobb角的上端椎和下端椎
    上端椎：倾斜角最大的椎体
    下端椎：倾斜角最小的椎体
    """
    if not spine.vertebrae:
        return None

    max_tilt = float('-inf')
    min_tilt = float('inf')
    upper_v = None
    lower_v = None

    for v in spine.vertebrae.values():
        tilt = calc_angle_from_points(v.top_left, v.top_right)
        if tilt > max_tilt:
            max_tilt = tilt
            upper_v = v
        if tilt < min_tilt:
            min_tilt = tilt
            lower_v = v

    return (upper_v, lower_v) if upper_v and lower_v else None


def _find_apex_vertebra_obj(spine: SpinePoints) -> Optional[VertebraPoints]:
    """找到顶椎对象"""
    if not spine.vertebrae:
        return None

    csvl_x = spine.csvl_x
    max_offset = 0
    apex = None

    for v in spine.vertebrae.values():
        offset = abs(v.center.x - csvl_x)
        if offset > max_offset:
            max_offset = offset
            apex = v

    return apex



# ============================================================
# 第六部分：使用示例
# ============================================================

def example_usage():
    """
    使用示例：从模型输出生成前端需要的JSON
    """
    from ultralytics import YOLO
    import json

    # 1. 加载模型
    pose_model = YOLO("runs/pose/train/weights/best.pt")
    pose_corner_model = YOLO("runs/pose_corner/train/weights/best.pt")

    # 2. 推理
    image_path = "test.jpg"
    pose_result = pose_model(image_path)
    pose_corner_result = pose_corner_model(image_path)

    # 3. 构建SpinePoints
    spine = build_spine_points(pose_result, pose_corner_result)

    # 4. 转换为前端格式
    frontend_json = convert_to_frontend_format(spine, image_id="IMG001")

    # 5. 保存
    with open("annotations_IMG001.json", "w") as f:
        json.dump(frontend_json, f, indent=2)

    print("生成的前端JSON:")
    print(json.dumps(frontend_json, indent=2))

    return frontend_json


if __name__ == "__main__":
    example_usage()

