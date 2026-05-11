#!/usr/bin/env python3
"""
可视化脚本：两层叠加
  Layer 1 (灰色细框): 所有检测到的椎体角点 V0, V1, V2, ...
  Layer 2 (彩色粗线): 各指标测量点 (T1 Tilt / Cobb / RSH / ...)

用法:
    python visualize_result.py --image spine.png --result result.json --out vis.png
"""
import json, argparse
import cv2
import numpy as np

# ── 指标颜色 (BGR) ──────────────────────────────────────────────────────────
MEASURE_COLORS = {
    "T1 Tilt": (255, 100,   0),   # 蓝
    "Cobb":    (  0, 200,   0),   # 绿
    "RSH":     (  0,   0, 220),   # 红
    "Pelvic":  (180,   0, 180),   # 紫
    "Sacral":  (  0, 180, 180),   # 青
    "AVT":     (  0, 165, 255),   # 橙
    "TS":      (255, 215,   0),   # 金
}
DEFAULT_COLOR = (200, 200, 200)

# ── 椎体角点颜色（按编号循环） ───────────────────────────────────────────────
VERT_PALETTE = [
    (160,160,160),(180,160,140),(140,180,160),(160,140,180),
    (180,180,140),(140,160,180),(180,140,160),(160,180,180),
]

def vert_color(rank):
    return VERT_PALETTE[rank % len(VERT_PALETTE)]

def text_with_bg(img, text, pt, font_scale=0.45, color=(255,255,255),
                 bg=(0,0,0), thickness=1):
    """带黑色背景的文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(pt[0]), int(pt[1])
    cv2.rectangle(img, (x-2, y-th-2), (x+tw+2, y+bl), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

# ── Layer 1: 所有椎体角点 ────────────────────────────────────────────────────
def draw_vertebrae(img, vertebrae, sx, sy):
    for vname, v in sorted(vertebrae.items(), key=lambda x: int(x[0][1:])):
        rank  = int(vname[1:])
        color = vert_color(rank)
        pts   = np.array([
            [v["tl"][0]*sx, v["tl"][1]*sy],
            [v["tr"][0]*sx, v["tr"][1]*sy],
            [v["br"][0]*sx, v["br"][1]*sy],
            [v["bl"][0]*sx, v["bl"][1]*sy],
        ], dtype=np.int32)
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 3, color, -1)
        cx = int(pts[:,0].mean()); cy = int(pts[:,1].mean())
        text_with_bg(img, vname, (cx-12, cy+5), font_scale=0.38,
                     color=(255,255,255), bg=color)

# ── Layer 2: 指标测量线 ──────────────────────────────────────────────────────
def draw_measurements(img, measurements, sx, sy):
    for m in measurements:
        mtype = m.get("type", "")
        pts   = m.get("points", [])
        color = MEASURE_COLORS.get(mtype, DEFAULT_COLOR)

        if mtype == "Cobb" and len(pts) == 4:
            # 上端椎终板线 + 下端椎终板线
            p = [(int(p["x"]*sx), int(p["y"]*sy)) for p in pts]
            cv2.line(img, p[0], p[1], color, 3, cv2.LINE_AA)
            cv2.line(img, p[2], p[3], color, 3, cv2.LINE_AA)
            for pt in p:
                cv2.circle(img, pt, 5, color, -1)
            mx = (p[0][0]+p[1][0])//2; my = (p[0][1]+p[1][1])//2
            text_with_bg(img, mtype, (mx+6, my-6), color=color, bg=(0,0,0))
        elif len(pts) >= 2:
            p1 = (int(pts[0]["x"]*sx), int(pts[0]["y"]*sy))
            p2 = (int(pts[1]["x"]*sx), int(pts[1]["y"]*sy))
            cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)
            cv2.circle(img, p1, 5, color, -1)
            cv2.circle(img, p2, 5, color, -1)
            mx = (p1[0]+p2[0])//2; my = (p1[1]+p2[1])//2
            text_with_bg(img, mtype, (mx+6, my-6), color=color, bg=(0,0,0))

# ── 图例 ─────────────────────────────────────────────────────────────────────
def draw_legend(img, measurements):
    y = 20
    for m in measurements:
        mtype = m.get("type","")
        color = MEASURE_COLORS.get(mtype, DEFAULT_COLOR)
        cv2.rectangle(img, (8, y-10), (22, y+4), color, -1)
        text_with_bg(img, mtype, (28, y+4), font_scale=0.45,
                     color=color, bg=(0,0,0))
        y += 22

# ── 主程序 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out",    default="result_vis.png")
    args = parser.parse_args()

    with open(args.result) as f:
        data = json.load(f)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    h, w = img.shape[:2]

    res_w = data.get("imageWidth",  w)
    res_h = data.get("imageHeight", h)
    sx = w / res_w if res_w else 1.0
    sy = h / res_h if res_h else 1.0

    # Layer 1: 所有椎体角点
    vertebrae = data.get("vertebrae", {})
    if vertebrae:
        draw_vertebrae(img, vertebrae, sx, sy)
    else:
        print("⚠️  JSON 中无 vertebrae 字段，请用新版 API 重新生成 result.json")

    # Layer 2: 指标测量线
    measurements = data.get("measurements", [])
    draw_measurements(img, measurements, sx, sy)
    draw_legend(img, measurements)

    cv2.imwrite(args.out, img)
    print(f"✅ 保存至: {args.out}")
    print(f"   检测椎体数: {len(vertebrae)}   指标数: {len(measurements)}")
