#!/usr/bin/env python3
"""
Spine Analysis API 测试脚本

使用方法:
    1. 先启动服务: cd 3-inference && python app.py
    2. 运行测试:   python example/test_api.py [--url http://localhost:8000]

对 example/ 目录下所有 *.png 逐一测试，每张图生成:
    - <stem>_result.json   推理结果
    - <stem>_vis.png       可视化叠加图（椎体角点 + 指标测量线）
"""
import requests, json, sys, subprocess, argparse
from pathlib import Path

HERE = Path(__file__).parent
VIS_SCRIPT = HERE / "visualize_result.py"


def check_health(base_url: str) -> bool:
    print("=" * 55)
    print("健康检查  GET /health")
    print("=" * 55)
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        d = r.json()
        print(f"  状态          : {d['status']}")
        print(f"  Pose 模型     : {'✅' if d['pose_model']        else '❌ 未加载'}")
        print(f"  PoseCorner 模型: {'✅' if d['pose_corner_model'] else '❌ 未加载'}")
        return d["status"] == "ok"
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接，请先启动服务: python app.py")
        return False
    except Exception as e:
        print(f"❌ {e}")
        return False


def predict_one(base_url: str, img_path: Path) -> dict | None:
    """调用 /predict，返回 JSON dict；失败返回 None"""
    with open(img_path, "rb") as f:
        r = requests.post(
            f"{base_url}/predict",
            files={"file": (img_path.name, f, "image/png")},
            params={"image_id": img_path.stem},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()


def print_summary(result: dict):
    n_vert   = len(result.get("vertebrae", {}))
    n_meas   = len(result.get("measurements", []))
    img_size = f"{result.get('imageWidth','?')} x {result.get('imageHeight','?')}"
    print(f"  图像尺寸  : {img_size} px")
    print(f"  检测椎体数: {n_vert}")
    if n_vert == 0:
        print("  ⚠️  vertebrae 字段为空，请确认使用最新版 app.py")
    print(f"  指标数    : {n_meas}")
    for m in result.get("measurements", []):
        pts_str = "  ".join(f"({p['x']:.0f},{p['y']:.0f})" for p in m["points"])
        print(f"    {m['type']:10s} {len(m['points'])}pt  {pts_str}")


def visualize(img_path: Path, json_path: Path, vis_path: Path):
    """调用 visualize_result.py 生成可视化图"""
    ret = subprocess.run(
        [sys.executable, str(VIS_SCRIPT),
         "--image", str(img_path),
         "--result", str(json_path),
         "--out",   str(vis_path)],
        capture_output=True, text=True
    )
    if ret.returncode == 0:
        print(f"  可视化    : {vis_path.name}")
    else:
        print(f"  ⚠️  可视化失败: {ret.stderr.strip()[-120:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000",
                        help="API 地址 (default: http://localhost:8000)")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print(f"\n🔬 Spine Analysis API 测试  ({base_url})\n")

    if not check_health(base_url):
        sys.exit(1)

    # 找 example/ 下所有 *.png
    images = sorted(HERE.glob("*.png"))
    if not images:
        print(f"\n❌ {HERE} 下没有找到 PNG 图片")
        sys.exit(1)

    print(f"\n共找到 {len(images)} 张图片: {[p.name for p in images]}\n")

    ok, fail = 0, 0
    for img_path in images:
        print(f"{'='*55}")
        print(f"POST /predict  →  {img_path.name}")
        print(f"{'='*55}")
        try:
            result   = predict_one(base_url, img_path)
            json_out = HERE / f"{img_path.stem}_result.json"
            vis_out  = HERE / f"{img_path.stem}_vis.png"

            with open(json_out, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  JSON 保存 : {json_out.name}")

            print_summary(result)
            visualize(img_path, json_out, vis_out)
            ok += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            fail += 1
        print()

    print("=" * 55)
    print(f"完成: {ok} 成功  {fail} 失败")
    print("=" * 55)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

