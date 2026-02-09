#!/usr/bin/env python3
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]


def draw_measurements(img, data):
    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size
    w_res = data.get("imageWidth", w_img)
    h_res = data.get("imageHeight", h_img)
    sx = w_img / w_res if w_res else 1.0
    sy = h_img / h_res if h_res else 1.0
    font = ImageFont.load_default()

    for i, m in enumerate(data.get("measurements", [])):
        pts = m.get("points", [])
        if len(pts) < 2:
            continue
        x1 = pts[0]["x"] * sx
        y1 = pts[0]["y"] * sy
        x2 = pts[1]["x"] * sx
        y2 = pts[1]["y"] * sy
        color = COLORS[i % len(COLORS)]
        draw.line([(x1, y1), (x2, y2)], fill=color, width=4)
        r = max(3, int(min(w_img, h_img) * 0.005))
        draw.ellipse([(x1 - r, y1 - r), (x1 + r, y1 + r)], fill=color)
        draw.ellipse([(x2 - r, y2 - r), (x2 + r, y2 + r)], fill=color)
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        label = m.get("type", "")
        # draw label background for readability
        #tw, th = draw.textsize(label, font=font)
        #pad = 4
        #bg_box = [mx - tw / 2 - pad, my - th / 2 - pad, mx + tw / 2 + pad, my + th / 2 + pad]
        #draw.rectangle(bg_box, fill=(0, 0, 0, 160))
        #ßdraw.text((mx - tw / 2, my - th / 2), label, fill="white", font=font)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize measurement results on an image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--result", required=True, help="Path to the JSON result file")
    parser.add_argument("--out", default="result_vis.png", help="Output image path")
    args = parser.parse_args()

    with open(args.result, "r") as f:
        data = json.load(f)

    img = Image.open(args.image).convert("RGB")
    draw_measurements(img, data)
    img.save(args.out)
    print("Saved:", args.out)
