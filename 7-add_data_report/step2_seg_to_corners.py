#!/usr/bin/env python3
"""
Step 2: 将 label_data 下的 Segmentation.seg.nrrd 转换为 YOLO pose-corner 格式标签
- 解析 seg.nrrd 获取每节椎体的 mask
- 转换为 YOLO polygon → 再提取 MinAreaRect 四角点 (TL/TR/BR/BL)
- 解剖名→类别ID 映射: C7→0, T1→1, ..., T12→12, L1→13, ..., L5→17
- 坐标系: NRRD(x,y) → DICOM 像素坐标，按 manifest.csv 中图像尺寸归一化

用法:
    conda activate cv
    python 7-add_data_report/step2_seg_to_corners.py

输出:
    7-add_data_report/output/labels/*.txt   YOLO pose-corner 格式
"""

import csv, sys
from pathlib import Path
import numpy as np
import nrrd
import cv2
from tqdm import tqdm

BASE_DIR   = Path(__file__).parent.parent
LABEL_DATA = BASE_DIR / "datasets/202605AP_PUMCH_Data/label_data"
MANIFEST   = Path(__file__).parent / "manifest.csv"
OUT_LABELS = Path(__file__).parent / "output/labels"

OUT_LABELS.mkdir(parents=True, exist_ok=True)

# 解剖名 → YOLO class_id  (同 seg_data 约定, C7=0 训练时留空)
ANAT_TO_CLS = {
    'C7': 0,
    'T1': 1,  'T2': 2,  'T3': 3,  'T4': 4,  'T5': 5,
    'T6': 6,  'T7': 7,  'T8': 8,  'T9': 9,  'T10': 10,
    'T11': 11, 'T12': 12,
    'L1': 13, 'L2': 14, 'L3': 15, 'L4': 16, 'L5': 17,
}


def sort_corners(box: np.ndarray) -> np.ndarray:
    """按 TL/TR/BR/BL 排序四角点"""
    top_idx    = np.argsort(box[:, 1])[:2]
    bottom_idx = np.argsort(box[:, 1])[2:]
    tl = top_idx[np.argmin(box[top_idx, 0])]
    tr = top_idx[np.argmax(box[top_idx, 0])]
    bl = bottom_idx[np.argmin(box[bottom_idx, 0])]
    br = bottom_idx[np.argmax(box[bottom_idx, 0])]
    return box[[tl, tr, br, bl]]


def parse_seg_nrrd(seg_path: Path):
    """返回 {class_id: mask_2d(H,W)} dict，坐标系与 NRRD x/y 一致"""
    data, header = nrrd.read(str(seg_path))
    # shape: (nx, ny, 1) → squeeze → (nx, ny)
    mask = data.squeeze()  # (nx, ny)

    # 解析 segment 名称
    label_to_cls = {}
    i = 0
    while True:
        name_key  = f'Segment{i}_Name'
        label_key = f'Segment{i}_LabelValue'
        if name_key not in header:
            break
        anat_name = header[name_key]
        label_val = int(header.get(label_key, -1))
        cls_id    = ANAT_TO_CLS.get(anat_name, -1)
        if cls_id >= 0:
            label_to_cls[label_val] = (cls_id, anat_name)
        i += 1

    nx, ny = mask.shape   # nx=cols_nrrd, ny=rows_nrrd
    result = {}
    for lv, (cls_id, name) in label_to_cls.items():
        m = (mask == lv).astype(np.uint8)
        # NRRD(nx,ny) → image(H=ny, W=nx): transpose
        result[cls_id] = (m.T, nx, ny)   # mask_hw, seg_w, seg_h
    return result


def mask_to_corners(mask_hw, seg_w, seg_h, img_w, img_h):
    """
    从 2D mask 提取 MinAreaRect 四角点，归一化到 img 坐标系
    mask_hw: HxW array (seg_h x seg_w)
    """
    contours, _ = cv2.findContours(mask_hw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None

    rect = cv2.minAreaRect(cnt.astype(np.float32))
    box  = cv2.boxPoints(rect)   # (4,2) in seg pixel coords

    # 缩放到 img 坐标系
    sx = img_w / seg_w
    sy = img_h / seg_h
    box[:, 0] *= sx
    box[:, 1] *= sy

    corners = sort_corners(box)  # TL TR BR BL in img pixels

    # 多边形轮廓（用于 bbox 计算）
    poly = cnt.squeeze().astype(np.float32)
    if poly.ndim == 1:
        poly = poly.reshape(1, 2)
    poly[:, 0] *= sx
    poly[:, 1] *= sy

    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    kpts = []
    for px, py in corners:
        kpts.extend([px / img_w, py / img_h, 2])

    return cx, cy, bw, bh, kpts


def main():
    if not MANIFEST.exists():
        sys.exit(f"❌ 找不到 {MANIFEST}，请先运行 step1_dicom_to_png.py")

    # 读取 manifest: {(pid, study_date): (png_name, img_w, img_h)}
    png_map = {}
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            png_map[(row['patient_id'], row['study_date'])] = (
                row['png_name'], int(row['width']), int(row['height'])
            )

    # 遍历 label_data
    seg_files = list(LABEL_DATA.rglob("Segmentation*.seg.nrrd"))
    print(f"找到 {len(seg_files)} 个分割文件，开始处理...")

    processed, skipped = 0, 0
    for seg_path in tqdm(seg_files, desc="seg→corners"):
        study_dir  = seg_path.parent
        study_date = study_dir.name
        pid        = study_dir.parent.name
        key = (pid, study_date)

        if key not in png_map:
            skipped += 1
            continue

        png_name, img_w, img_h = png_map[key]
        stem = Path(png_name).stem
        out_txt = OUT_LABELS / f"{stem}.txt"

        try:
            masks = parse_seg_nrrd(seg_path)
            lines = []
            for cls_id in sorted(masks.keys()):
                mask_hw, seg_w, seg_h = masks[cls_id]
                res = mask_to_corners(mask_hw, seg_w, seg_h, img_w, img_h)
                if res is None:
                    continue
                cx, cy, bw, bh, kpts = res
                kpt_str = " ".join(f"{v:.6f}" for v in kpts)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpt_str}")

            with open(out_txt, 'w') as f:
                f.write("\n".join(lines))
            processed += 1
        except Exception as e:
            print(f"\n⚠️  {seg_path}: {e}")
            skipped += 1

    print(f"\n✅ 完成: {processed} 张标签生成, {skipped} 个跳过/错误")
    print(f"   输出目录: {OUT_LABELS}")


if __name__ == "__main__":
    main()
