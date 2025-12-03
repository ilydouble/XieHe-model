import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 训练实例分割模型
    model = YOLO("yolo11n-seg.yaml").load("../weights/yolo11n-seg.pt")
    
    model.train(
        data='../seg_data/dataset.yaml',  # 分割数据集配置
        imgsz=640,
        epochs=300,
        batch=64,
        workers=9,
        device='0',
        optimizer='SGD',
        project='../runs/seg',
        name='train',
    )
