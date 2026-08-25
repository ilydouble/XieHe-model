#!/usr/bin/env python3
"""检查横向大图的黑边情况"""
import numpy as np
import os
from PIL import Image

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'medical-image-files')
out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_border_result.txt')
import sys
fh = open(out_file, 'w')
def log(*a): s = ' '.join(str(x) for x in a); print(s); fh.write(s+'\n')

threshold = 10

targets = [f for f in sorted(os.listdir(img_dir))
           if '正位' in f and not ('_nxy_' in f or '_projected_' in f)]

log(f'找到 {len(targets)} 张图像\n')

for fname in targets:
    path = os.path.join(img_dir, fname)
    try:
        pil = Image.open(path).convert('L')   # 转灰度
    except Exception as e:
        log(f'{fname}: 读取失败 {e}')
        continue

    img = np.array(pil)
    h, w = img.shape

    left   = next((c for c in range(w)     if img[:, c].mean() > threshold), 0)
    right  = next((c for c in range(w-1,-1,-1) if img[:, c].mean() > threshold), w-1) + 1
    top    = next((r for r in range(h)     if img[r, :].mean() > threshold), 0)
    bottom = next((r for r in range(h-1,-1,-1) if img[r, :].mean() > threshold), h-1) + 1

    crop_w = right - left
    crop_h = bottom - top
    ratio  = crop_w / crop_h if crop_h > 0 else 0
    orient = '竖向 ✅' if ratio < 1 else '横向 ⚠️'

    log(f'{fname}')
    log(f'  原始 {w}x{h}  ->  去黑边后 {crop_w}x{crop_h}  '
        f'宽高比 {ratio:.2f}  {orient}')
    log(f'  黑边: 左={left}px 右={w-right}px 上={top}px 下={h-bottom}px')
    log('')

fh.close()
print(f'结果已写入 {out_file}')
