# Adapted from https://medium.com/@lfoster49203/developing-real-time-object-detection-using-yolov8-and-custom-datasets-db01ea580c9c

import cv2
from ultralytics import YOLO

# *************************
from time import time
# *************************

import argparse

parser = argparse.ArgumentParser(description='Runs the Yolo detect operation on a video stream from an mp4 file.')
parser.add_argument('--model', help='The YOLO model to use when processing the video stream.', default='yolo8n_ncnn')
parser.add_argument('--video', help='The short name of the video file to analyze.', default='hailo_hd')
args = parser.parse_args()

yolo_models = {'yolo8n':'/home/rpi5/Desktop/Yolo_Models/detect/yolov8n.pt',
               'yolo8s':'/home/rpi5/Desktop/Yolo_Models/detect/yolov8s.pt',
               'yolo8m':'/home/rpi5/Desktop/Yolo_Models/detect/yolov8m.pt',
               'yolo8n_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolov8n_ncnn_model',
               'yolo8s_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolov8s_ncnn_model',
               'yolo8m_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolov8m_ncnn_model',
               'yolo11n':'/home/rpi5/Desktop/Yolo_Models/detect/yolo11n.pt',
               'yolo11s':'/home/rpi5/Desktop/Yolo_Models/detect/yolo11s.pt',
               'yolo11m':'/home/rpi5/Desktop/Yolo_Models/detect/yolo11m.pt',
               'yolo11n_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolo11n_ncnn_model',
               'yolo11s_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolo11s_ncnn_model',
               'yolo11m_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolo11m_ncnn_model'
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

videos = {'hailo_hd':'/home/rpi5/Desktop/Yolo_Demos/videos/example.mp4',
          'london_sd':'/home/rpi5/Desktop/Yolo_Demos/videos/2954065-sd_960_540_30fps.mp4',
          'london_hd':'/home/rpi5/Desktop/Yolo_Demos/videos/2954065-hd_1280_720_30fps.mp4',
          'london_full_hd':'/home/rpi5/Desktop/Yolo_Demos/videos/2954065-hd_1920_1080_30fps.mp4',
          'paris_sd':'/home/rpi5/Desktop/Yolo_Demos/videos/4979729-sd_960_540_30fps.mp4',
          'paris_hd':'/home/rpi5/Desktop/Yolo_Demos/videos/4979729-hd_1280_720_30fps.mp4',
          'paris_full_hd':'/home/rpi5/Desktop/Yolo_Demos/videos/4979729-hd_1920_1080_30fps.mp4',
          'nyc_sd':'/home/rpi5/Desktop/Yolo_Demos/videos/12127639_1160_540_30fps.mp4',
          'nyc_hd':'/home/rpi5/Desktop/Yolo_Demos/videos/12127640_1546_720_30fps.mp4',
          'nyc_full_hd':'/home/rpi5/Desktop/Yolo_Demos/videos/12127641_2320_1080_30fps.mp4'
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
    results = model(frame)
    
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
