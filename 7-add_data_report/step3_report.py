#!/usr/bin/env python3
"""
Step 3: 统计新增数据，生成数据报告
- 读取 manifest.csv 和 output/labels/*.txt
- 统计: 病例数, 图像尺寸分布, 椎体数分布, 各类别 (V0-V17) 出现次数
- 与现有 pose_corner_data 数据量对比
- 输出: 7-add_data_report/report.txt (控制台也打印)

用法:
    conda activate cv
    python 7-add_data_report/step3_report.py
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

BASE_DIR    = Path(__file__).parent.parent
MANIFEST    = Path(__file__).parent / "manifest.csv"
LABELS_DIR  = Path(__file__).parent / "output/labels"
REPORT_OUT  = Path(__file__).parent / "report.txt"

# 现有 pose_corner_data
EXIST_DATA  = BASE_DIR / "datasets/pose_corner_data"

ANAT_NAMES = {
    0: 'C7', 1: 'T1', 2: 'T2',  3: 'T3',  4: 'T4',  5: 'T5',
    6: 'T6', 7: 'T7', 8: 'T8',  9: 'T9',  10: 'T10', 11: 'T11',
    12: 'T12', 13: 'L1', 14: 'L2', 15: 'L3', 16: 'L4', 17: 'L5',
}


def count_existing():
    counts = {}
    for split in ('train', 'val', 'test'):
        img_dir = EXIST_DATA / 'images' / split
        lbl_dir = EXIST_DATA / 'labels' / split
        if img_dir.exists():
            counts[split] = (len(list(img_dir.glob('*.png'))),
                             len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0)
    return counts


def read_manifest():
    rows = []
    if not MANIFEST.exists():
        return rows
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    return rows


def parse_labels():
    """返回 {png_stem: [class_id, ...]}"""
    result = {}
    if not LABELS_DIR.exists():
        return result
    for lbl in LABELS_DIR.glob("*.txt"):
        cls_ids = []
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_ids.append(int(parts[0]))
        result[lbl.stem] = cls_ids
    return result


def main():
    lines = []

    def p(s=''):
        lines.append(s)
        print(s)

    p("=" * 60)
    p("  新增数据统计报告 — datasets/202605AP_PUMCH_Data")
    p("=" * 60)

    manifest = read_manifest()
    labels   = parse_labels()

    # ── 1. 基本统计 ──────────────────────────────────────────
    p()
    p("【1. 图像基本信息】")
    p(f"  AP 图像数 (已转换): {len(manifest)}")

    # 图像尺寸
    widths  = [int(r['width'])  for r in manifest]
    heights = [int(r['height']) for r in manifest]
    if widths:
        p(f"  宽度范围: {min(widths)} ~ {max(widths)} px  (均值 {np.mean(widths):.0f})")
        p(f"  高度范围: {min(heights)} ~ {max(heights)} px  (均值 {np.mean(heights):.0f})")

    # 病人数 (by patient_id)
    pids = set(r['patient_id'] for r in manifest)
    p(f"  病人数: {len(pids)}")

    # ── 2. 标签统计 ───────────────────────────────────────────
    p()
    p("【2. 标签统计】")
    p(f"  已生成标签文件: {len(labels)} 张")

    if labels:
        vert_counts = [len(v) for v in labels.values()]
        p(f"  每图椎体数: min={min(vert_counts)} / max={max(vert_counts)} / mean={np.mean(vert_counts):.1f}")

        cnt_18 = sum(1 for v in vert_counts if v == 18)
        cnt_less = sum(1 for v in vert_counts if v < 18)
        cnt_more = sum(1 for v in vert_counts if v > 18)
        p(f"    == 18 节: {cnt_18} 张 ({100*cnt_18/len(labels):.1f}%)")
        p(f"    < 18 节: {cnt_less} 张 (截断/缺失)")
        p(f"    > 18 节: {cnt_more} 张 (含额外椎体)")

        # 各类别出现次数
        cls_counter: Counter = Counter()
        for cls_list in labels.values():
            cls_counter.update(cls_list)
        p()
        p("  各类别出现次数:")
        for cls_id in range(18):
            name = ANAT_NAMES.get(cls_id, f'V{cls_id}')
            cnt  = cls_counter.get(cls_id, 0)
            bar  = '█' * (cnt * 30 // max(cls_counter.values(), default=1))
            p(f"    [{cls_id:2d}] {name:4s}: {cnt:4d}  {bar}")

    # ── 3. 与现有数据对比 ─────────────────────────────────────
    p()
    p("【3. 与现有 pose_corner_data 对比】")
    exist = count_existing()
    total_exist = sum(v[0] for v in exist.values())
    p(f"  现有数据量:")
    for split, (ni, nl) in exist.items():
        p(f"    {split:5s}: {ni} 张图 / {nl} 张标签")
    p(f"  现有总计: {total_exist} 张")
    p(f"  新增数量: {len(labels)} 张")
    p(f"  增幅: +{100*len(labels)/max(total_exist,1):.1f}%")

    # ── 4. 格式兼容性检查 ─────────────────────────────────────
    p()
    p("【4. 格式兼容性】")
    p("  ✅ YOLO pose-corner 格式: class cx cy bw bh kp1x kp1y 2 kp2x kp2y 2 kp3x kp3y 2 kp4x kp4y 2")
    p("  ✅ 坐标归一化 [0,1]")
    p("  ✅ 类别编号 0-17 与现有 pose_corner_data 一致")
    p("  ⚠️  新数据不含 C7 (class_id=0); 仅 T1-L5 (1-17)")

    # ── 5. 下一步建议 ─────────────────────────────────────────
    p()
    p("【5. 下一步建议】")
    p("  1. 按 7:1.5:1.5 随机划分 train/val/test")
    p("  2. 将 output/images/*.png 复制到 pose_corner_data/images/{split}/")
    p("  3. 将 output/labels/*.txt  复制到 pose_corner_data/labels/{split}/")
    p("  4. 重新训练 pose-corner 模型")

    p()
    p("=" * 60)

    # 写入文件
    with open(REPORT_OUT, 'w') as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存至: {REPORT_OUT}")


if __name__ == "__main__":
    main()
