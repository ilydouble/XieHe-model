import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO('../runs/seg/train/weights/best.pt')  # 或者 'last.pt'

    metrics = model.val(
        data='../seg_data/dataset.yaml',
        # split='test',  # 使用测试集
        batch=16,
        imgsz=640,
        device='0',
        project='../runs/seg',
        name='eval',
    )

    # 打印主要指标
    print("\n========== 分割评估指标 ==========")
    # Box (检测) 汇总
    print(f"Box mAP50:      {metrics.box.map50:.4f}")
    print(f"Box mAP75:      {metrics.box.map75:.4f}")
    print(f"Box mAP50-95:   {metrics.box.map:.4f}")
    print(f"Box mean P(mp): {metrics.box.mp:.4f}")
    print(f"Box mean R(mr): {metrics.box.mr:.4f}")
    # 分割汇总
    print("--------------------------------------")
    print(f"Mask mAP50:      {metrics.seg.map50:.4f}")
    print(f"Mask mAP75:      {metrics.seg.map75:.4f}")
    print(f"Mask mAP50-95:   {metrics.seg.map:.4f}")
    print(f"Mask mean P(mp): {metrics.seg.mp:.4f}")
    print(f"Mask mean R(mr): {metrics.seg.mr:.4f}")
