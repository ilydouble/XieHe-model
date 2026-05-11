#!/usr/bin/env python3
"""探索新数据格式，输出基本信息"""
import nrrd
import numpy as np

BASE = "/Users/liruirui/Documents/code/spine/Model/datasets/202605AP_PUMCH_Data"

# 1. 检查图像 NRRD
img_nrrd = f"{BASE}/label_data/2580571733/CR_TSPINE_20220926/1 AP AEC.nrrd"
data, header = nrrd.read(img_nrrd)
print("=== Image NRRD ===")
print("Shape:", data.shape, "dtype:", data.dtype)
print("Range:", data.min(), "-", data.max())
print("Space:", header.get("space"))
print("Space directions:", header.get("space directions"))
print("Space origin:", header.get("space origin"))

# 2. 检查分割 NRRD
seg_nrrd = f"{BASE}/label_data/2580571733/CR_TSPINE_20220926/Segmentation.seg.nrrd"
data2, header2 = nrrd.read(seg_nrrd)
print("\n=== Segmentation NRRD ===")
print("Shape:", data2.shape, "dtype:", data2.dtype)
print("Unique labels:", np.unique(data2))
print("Space:", header2.get("space"))
print("Space directions:", header2.get("space directions"))
print("Space origin:", header2.get("space origin"))

# 获取所有 segment 名称
segments = {}
for k, v in header2.items():
    if k.endswith("_Name") and "Segment" in k:
        seg_id = k.rsplit("_", 1)[0]
        label_key = seg_id + "_LabelValue"
        label_val = header2.get(label_key, "?")
        segments[label_val] = v
print("Segments (label->name):", dict(sorted(
    segments.items(), key=lambda x: (int(x[0]) if str(x[0]).isdigit() else 99)
)))
