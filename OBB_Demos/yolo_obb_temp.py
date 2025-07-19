from ultralytics import YOLO

# Load a pretrained YOLO11n model
#model = YOLO('/home/rpi5/Desktop/Yolo_Models/obb/NCNN_1024/yolov8n-obb_ncnn_model')
model = YOLO('/home/rpi5/Desktop/Yolo_Models/obb/yolov8n-obb.pt')

# Define path to video file
source = '/home/rpi5/Desktop/Yolo_Demos/OBB_videos/8918404-hd_1280_720_30fps.mp4'

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects