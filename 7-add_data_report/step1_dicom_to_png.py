#!/usr/bin/env python3
"""
Step 1: 将 x_data 下的 DICOM 文件转换为 PNG
- 仅处理 AP (前后位) 图像 (SeriesDescription 含 'AP')
- 16-bit DICOM → 8-bit PNG (percentile 归一化)
- 输出文件名: {SOP_Instance_UID}.png
- 同时生成 manifest.csv 记录对应关系

用法:
    conda activate cv
    python 7-add_data_report/step1_dicom_to_png.py

输出:
    7-add_data_report/output/images/*.png
    7-add_data_report/manifest.csv
"""

import os, csv, glob
from pathlib import Path
import numpy as np
import cv2
import pydicom
from tqdm import tqdm

BASE_DIR  = Path(__file__).parent.parent  # Model/
X_DATA    = BASE_DIR / "datasets/202605AP_PUMCH_Data/x_data"
OUT_IMG   = Path(__file__).parent / "output/images"
MANIFEST  = Path(__file__).parent / "manifest.csv"

OUT_IMG.mkdir(parents=True, exist_ok=True)


def dcm_to_uint8(dcm: pydicom.Dataset) -> np.ndarray:
    """将 DICOM 像素数组转换为 8-bit 灰度图，自动处理光度解释和窗宽窗位"""
    arr = dcm.pixel_array.astype(np.float32)

    # 尝试使用 DICOM 窗宽/窗位
    wc = float(getattr(dcm, 'WindowCenter', 0) or 0)
    ww = float(getattr(dcm, 'WindowWidth',  0) or 0)
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])

    if ww > 0:
        lo = wc - ww / 2
        hi = wc + ww / 2
    else:
        # 回退: 1%-99% 百分位
        lo = float(np.percentile(arr, 1))
        hi = float(np.percentile(arr, 99))

    arr = np.clip((arr - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

    # MONOCHROME1: 高值=低密度，需要反转
    photo = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
    if isinstance(photo, str) and 'MONOCHROME1' in photo:
        arr = 255 - arr

    return arr


def find_dcm_files(x_data: Path):
    """遍历 x_data，返回所有 DICOM 文件路径（按病人+检查日期分组）"""
    records = []
    for pid_dir in sorted(x_data.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        for study_dir in sorted(pid_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            study_date = study_dir.name
            dcm_files = list(study_dir.rglob("*.dcm"))
            for dcm_path in dcm_files:
                records.append((pid, study_date, dcm_path))
    return records


def main():
    records = find_dcm_files(X_DATA)
    print(f"找到 {len(records)} 个 DICOM 文件，开始处理...")

    manifest_rows = []
    skipped_lat  = 0
    converted    = 0
    errors       = 0

    for pid, study_date, dcm_path in tqdm(records, desc="DICOM→PNG"):
        try:
            dcm = pydicom.dcmread(str(dcm_path))
            series_desc = str(getattr(dcm, 'SeriesDescription', '')).upper()

            # 只处理 AP 图像
            if 'AP' not in series_desc and 'ANTERIOR' not in series_desc:
                skipped_lat += 1
                continue

            sop_uid = str(getattr(dcm, 'SOPInstanceUID', dcm_path.stem))
            out_png = OUT_IMG / f"{sop_uid}.png"

            if out_png.exists():
                # 已转换，跳过
                h, w = cv2.imread(str(out_png)).shape[:2]
                manifest_rows.append([pid, study_date, str(dcm_path.relative_to(BASE_DIR)),
                                      out_png.name, w, h, series_desc])
                converted += 1
                continue

            arr = dcm_to_uint8(dcm)
            h, w = arr.shape[:2]
            cv2.imwrite(str(out_png), arr)

            manifest_rows.append([pid, study_date, str(dcm_path.relative_to(BASE_DIR)),
                                   out_png.name, w, h, series_desc])
            converted += 1
        except Exception as e:
            print(f"\n⚠️  {dcm_path}: {e}")
            errors += 1

    # 写 manifest
    with open(MANIFEST, 'w', newline='') as f:
        w_csv = csv.writer(f)
        w_csv.writerow(['patient_id', 'study_date', 'dicom_path',
                        'png_name', 'width', 'height', 'series_desc'])
        w_csv.writerows(manifest_rows)

    print(f"\n✅ 转换完成: {converted} 张 AP 图, {skipped_lat} 张非AP跳过, {errors} 个错误")
    print(f"   输出目录: {OUT_IMG}")
    print(f"   清单文件: {MANIFEST}")


if __name__ == "__main__":
    main()
