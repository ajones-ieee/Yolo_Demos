# Adapted from https://medium.com/@lfoster49203/developing-real-time-object-detection-using-yolov8-and-custom-datasets-db01ea580c9c

import cv2
from ultralytics import YOLO

# *************************
from time import time
# *************************

import argparse

parser = argparse.ArgumentParser(description='Runs the Yolo OBB operation on a video stream from an mp4 file.')
parser.add_argument('--model', help='The YOLO model to use when processing the video stream.', default='yolo8n-obb_ncnn')
parser.add_argument('--video', help='The short name of the video file to analyze.', default='roundabout_sd')
args = parser.parse_args()

yolo_models = {'yolo8n-obb':'/home/rpi5/Desktop/Yolo_Models/obb/yolov8n-obb.pt',
               'yolo8s-obb':'/home/rpi5/Desktop/Yolo_Models/obb/yolov8s-obb.pt',
               'yolo8m-obb':'/home/rpi5/Desktop/Yolo_Models/obb/yolov8m-obb.pt',
               'yolo8n-obb_ncnn':'/home/rpi5/Desktop/Yolo_Models/obb/NCNN_1024/yolov8n-obb_ncnn_model',
               'yolo8s-obb_ncnn':'/home/rpi5/Desktop/Yolo_Models/obb/NCNN_1024/yolov8s-obb_ncnn_model',
                'yolo11n-obb':'/home/rpi5/Desktop/Yolo_Models/obb/yolo11n-obb.pt',
               'yolo11s-obb':'/home/rpi5/Desktop/Yolo_Models/obb/yolo11s-obb.pt',
               'yolo11m-obb':'/home/rpi5/Desktop/Yolo_Models/obb/yolo11m-obb.pt',
               'yolo11n-obb_ncnn':'/home/rpi5/Desktop/Yolo_Models/obb/NCNN_1024/yolo11n-obb_ncnn_model',
               'yolo11s-obb_ncnn':'/home/rpi5/Desktop/Yolo_Models/obb/NCNN_1024/yolo11s-obb_ncnn_model',
               'yolo11m-obb_ncnn':'/home/rpi5/Desktop/Yolo_Models/obb/NCNN_1024/yolo11m-obb_ncnn_model'
               }

try:
    yolo_model = yolo_models[args.model]
except KeyError:
    print('Invalid value of', args.model, 'for model. Valid values are:')
    for yolo_model in list(yolo_models.keys()):
        print(yolo_model)
    exit(1)
model = YOLO(yolo_model)
print ('Using model ' + yolo_model)

videos = {'traffic1_sd':'/home/rpi5/Desktop/Yolo_Demos/OBB_videos/7005467-sd_960_540_30fps.mp4',
          'traffic1_hd':'/home/rpi5/Desktop/Yolo_Demos/OBB_videos/7005467-hd_1280_720_30fps.mp4',
          'traffic1_full_hd':'/home/rpi5/Desktop/Yolo_Demos/OBB_videos/7005467-hd_1920_1080_30fps.mp4',
          'roundabout_sd':'/home/rpi5/Desktop/Yolo_Demos/OBB_videos/8918404-sd_960_540_30fps.mp4',
          'harbor1_sd':'/home/rpi5/Desktop/Yolo_Demos/OBB_videos/9301986-sd_960_540_30fps.mp4',
          'harbor1_hd':'/home/rpi5/Desktop/Yolo_Demos/OBB_videos/9301986-hd_1280_720_30fps.mp4',
          'harbor1_full_hd':'/home/rpi5/Desktop/Yolo_Demos/OBB_videos/9301986-hd_1920_1080_30fps.mp4'
         }

try:
    video = videos[args.video]
except KeyError:
    print ('Invalid value of', args.video, 'for videos. Valid values are:')
    for video in list(videos.keys()):
        print(video)
    exit(1)

print ('to analyze video ' + video)

# Open a video stream (webcam)
cap = cv2.VideoCapture(video)

title = 'Model = '+ args.model + ', Video = ' + args.video

loop_time = time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame, imgsz=1024)
    
    current_time = time()
    fps = 1 / (current_time - loop_time)
    loop_time = current_time

    # Visualize results on the frame
    annotated_frame = results[0].plot()  # Draw bounding boxes on the frame
    
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow(title, annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
