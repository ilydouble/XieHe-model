import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import cv2
import os


def segment_image(model_path, image_path):
    """
    对单张图像进行实例分割并输出结果

    Args:
        model_path: 训练好的模型权重路径
        image_path: 输入图像路径

    Returns:
        results: YOLO推理结果对象
    """
    model = YOLO(model_path)

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.5,
        save=False,
        verbose=False,
    )

    return results


def extract_masks(results):
    """
    从推理结果中提取分割掩码信息

    Returns:
        list: 包含所有检测对象的分割信息
    """
    all_detections = []

    for result in results:
        if result.masks is None or len(result.masks) == 0:
            print("⚠️  未检测到任何对象")
            continue

        masks = result.masks
        boxes = result.boxes

        for i in range(len(masks)):
            detection = {
                'class_id': int(boxes.cls[i].cpu().numpy()),
                'class_name': result.names[int(boxes.cls[i].cpu().numpy())],
                'confidence': float(boxes.conf[i].cpu().numpy()),
                'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                'mask_polygon': masks.xy[i].tolist(),  # 多边形格式
                'mask_polygon_normalized': masks.xyn[i].tolist(),  # 归一化多边形
            }
            all_detections.append(detection)

    return all_detections


def print_results(detections):
    """
    打印分割结果信息
    """
    print("\n" + "="*60)
    print("🎯 分割检测结果")
    print("="*60)

    if not detections:
        print("未检测到任何对象")
        return

    for i, det in enumerate(detections):
        print(f"\n对象 {i + 1}:")
        print(f"  类别: {det['class_name']} (ID: {det['class_id']})")
        print(f"  置信度: {det['confidence']:.3f}")
        print(f"  边界框: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
        print(f"  掩码点数: {len(det['mask_polygon'])}")


def visualize_segmentation(image_path, results, output_path):
    """
    可视化分割结果
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    # 使用 YOLO 的内置可视化
    annotated = results[0].plot()

    cv2.imwrite(output_path, annotated)
    print(f"\n🎨 可视化图像已保存")

    return annotated


if __name__ == '__main__':
    # ========== 配置参数（请根据实际情况修改） ==========
    MODEL_PATH = '../runs/seg/train/weights/best.pt'  # 模型路径
    IMAGE_PATH = '../seg_data/images/test/example.png'  # 输入图像路径
    OUTPUT_DIR = '../runs/seg/inference'  # 输出目录

    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        exit(1)

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 图像文件不存在: {IMAGE_PATH}")
        print(f"💡 请修改 IMAGE_PATH 为实际的图像路径")
        exit(1)

    # 进行推理
    print(f"🔍 正在处理图像: {IMAGE_PATH}")
    results = segment_image(MODEL_PATH, IMAGE_PATH)

    # 提取分割结果
    detections = extract_masks(results)

    # 打印结果
    print_results(detections)

    # 可视化
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_basename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_image_path = os.path.join(OUTPUT_DIR, f'{image_basename}_segmented.jpg')
    visualize_segmentation(IMAGE_PATH, results, output_image_path)

    print("\n✅ 推理完成！可视化结果已保存到:")
    print(f"   {output_image_path}")