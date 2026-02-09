#!/usr/bin/env python3
"""
Spine Analysis API 测试脚本

使用方法:
    1. 先启动服务: cd 3-inference && python app.py
    2. 运行测试: python example/test_api.py

依赖:
    pip install requests
"""

import requests
import json
import sys
from pathlib import Path

# API 配置
API_BASE_URL = "http://localhost:8888"
TEST_IMAGE = Path(__file__).parent / "test_spine1.png"
OUTPUT_FILE = Path(__file__).parent / "result.json"


def test_health():
    """测试健康检查接口"""
    print("=" * 50)
    print("测试健康检查接口: GET /health")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        result = response.json()
        print(f"状态: {result['status']}")
        print(f"Pose 模型: {'✅ 已加载' if result['pose_model'] else '❌ 未加载'}")
        print(f"Pose Corner 模型: {'✅ 已加载' if result['pose_corner_model'] else '❌ 未加载'}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保服务已启动: python app.py")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_predict():
    """测试推理接口"""
    print("\n" + "=" * 50)
    print("测试推理接口: POST /predict")
    print("=" * 50)
    
    if not TEST_IMAGE.exists():
        print(f"❌ 测试图片不存在: {TEST_IMAGE}")
        return False
    
    print(f"测试图片: {TEST_IMAGE}")
    print(f"图片大小: {TEST_IMAGE.stat().st_size / 1024:.1f} KB")
    
    try:
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": (TEST_IMAGE.name, f, "image/png")}
            params = {"image_id": "TEST_001"}
            response = requests.post(
                f"{API_BASE_URL}/predict",
                files=files,
                params=params
            )
        
        response.raise_for_status()
        result = response.json()
        
        # 保存结果
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果已保存到: {OUTPUT_FILE}")
        
        # 打印结果摘要
        print(f"\n图片ID: {result['imageId']}")
        print(f"检测到的测量项: {len(result['measurements'])} 个")
        print("\n测量项列表:")
        print("-" * 40)
        
        for m in result["measurements"]:
            points_str = ", ".join([f"({p['x']:.1f}, {p['y']:.1f})" for p in m["points"]])
            print(f"  {m['type']:12s} | {len(m['points'])} 点 | {points_str[:50]}...")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保服务已启动: python app.py")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n🔬 Spine Analysis API 测试\n")
    
    # 测试健康检查
    if not test_health():
        sys.exit(1)
    
    # 测试推理
    if not test_predict():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()

