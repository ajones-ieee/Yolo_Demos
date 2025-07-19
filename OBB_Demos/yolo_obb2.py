from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-obb.pt')

results = model('/home/rpi5/Desktop/old_yolo_demo/videos/9301986-hd_1280_720_30fps.mp4',
                show=True, save=False)