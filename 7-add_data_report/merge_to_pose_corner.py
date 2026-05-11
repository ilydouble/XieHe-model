#!/usr/bin/env python3
"""
将 7-add_data_report/output/ 新增数据合并到 datasets/pose_corner_data/
并对全量数据重新按 7:1.5:1.5 随机划分 train/val/test

用法:
    python 7-add_data_report/merge_to_pose_corner.py
"""

import random
import shutil
import tempfile
from pathlib import Path

random.seed(42)

BASE_DIR   = Path(__file__).parent.parent
DST        = BASE_DIR / "datasets/pose_corner_data"
NEW_IMG    = Path(__file__).parent / "output/images"
NEW_LBL    = Path(__file__).parent / "output/labels"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# test = 1 - train - val = 0.10


def collect_existing():
    """收集现有所有 stem（跨 train/val/test）"""
    stems = {}
    for split in ("train", "val", "test"):
        img_dir = DST / "images" / split
        lbl_dir = DST / "labels" / split
        for img in img_dir.glob("*.png"):
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                stems[img.stem] = (img, lbl)
    return stems


def collect_new():
    """收集新增数据 stem"""
    stems = {}
    for img in NEW_IMG.glob("*.png"):
        lbl = NEW_LBL / (img.stem + ".txt")
        if lbl.exists():
            stems[img.stem] = (img, lbl)
    return stems


def main():
    existing = collect_existing()
    new_data  = collect_new()

    overlap = set(existing) & set(new_data)
    if overlap:
        print(f"⚠️  发现 {len(overlap)} 个重复 stem，新数据优先覆盖")

    all_data = {**existing, **new_data}   # 新数据覆盖同名旧数据
    stems    = sorted(all_data.keys())
    random.shuffle(stems)

    n       = len(stems)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits  = {
        "train": stems[:n_train],
        "val"  : stems[n_train:n_train + n_val],
        "test" : stems[n_train + n_val:],
    }

    print(f"总计 {n} 张  →  train:{len(splits['train'])}  val:{len(splits['val'])}  test:{len(splits['test'])}")

    # 先把所有源文件暂存到临时目录（防止源目录和目标目录重叠）
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "images").mkdir()
        (tmp / "labels").mkdir()
        for stem, (src_img, src_lbl) in all_data.items():
            shutil.copy2(src_img, tmp / "images" / src_img.name)
            shutil.copy2(src_lbl, tmp / "labels" / (stem + ".txt"))

        # 清空旧目录
        for split in ("train", "val", "test"):
            for d in (DST / "images" / split, DST / "labels" / split):
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)

        # 从临时目录按 split 分发
        for split, stem_list in splits.items():
            img_out = DST / "images" / split
            lbl_out = DST / "labels" / split
            for stem in stem_list:
                src_img, src_lbl = all_data[stem]
                shutil.copy2(tmp / "images" / src_img.name, img_out / src_img.name)
                shutil.copy2(tmp / "labels" / (stem + ".txt"), lbl_out / (stem + ".txt"))

    print("✅  合并完成！")
    print(f"   数据集位置: {DST}")


if __name__ == "__main__":
    main()
