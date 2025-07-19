import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

import argparse

parser = argparse.ArgumentParser(description='Runs the Yolo detect operation on a video stream from a camera.')
parser.add_argument('--model', help='The YOLO model to use when processing the video stream.', default='yolo8n_ncnn')
args = parser.parse_args()

yolo_models = {'yolo5n':'/home/rpi5/Desktop/Yolo_Models/detect/yolov5nu.pt',
               'yolo5s':'/home/rpi5/Desktop/Yolo_Models/detect/yolov5su.pt',
               'yolo5m':'/home/rpi5/Desktop/Yolo_Models/detect/yolov5mu.pt',
               'yolo5n_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolov5nu_ncnn_model',
               'yolo5s_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolov5su_ncnn_model',
               'yolo5m_ncnn':'/home/rpi5/Desktop/Yolo_Models/detect/NCNN_640/yolov5mu_ncnn_model',
               'yolo8n':'/home/rpi5/Desktop/Yolo_Models/detect/yolov8n.pt',
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


# Set up the camera with Picam

# The global variable returned by Picamera2.global_camera_info() will return a list of
# cameras attached to the Raspberry Pi. By default, we will always use the first camera
# in that list. Any cameras connected by the CSI connector will always be listed before
# any of the cameras that are connected via USB. Beyond that, the order in which the USB
# cameras are listed is not guaranteed. To tell whether a particular camera is a CSI-
# connected camera or a USB-connected camera, you can look at the Id string associated
# with the camera. It will contain 'i2c' as part of the string if it is a CSI-connected
# camera and 'usb' if it is a USB-connected camera. This information may be needed because
# the Pi Camera Module 3 uses RGB888 format while many USB cameras use 'BGR888 format.
camera_list = Picamera2.global_camera_info()
if len(camera_list) == 0:
    print ('No camera detected.')
    exit()
elif len(camera_list) == 1:
    camera_instance = 0
else:
    print('\nThe following cameras were detected:')
    camera_num = 1
    for camera in camera_list:
        if camera['Model'] == 'imx708':
            camera_model = 'Raspberry Pi Camera Module'
        else:
            camera_model = camera['Model']
        print(camera_num, camera_model)
        camera_num = camera_num + 1
    try:
        camera_instance = int(input('Enter the number for the camera that you wish to use: '))
    except:
        print('Invalid camera number.')
        exit()
    if (camera_instance > len(camera_list)):
        print('Invalid camera number.')
        exit()
    else:
        camera_instance = camera_instance - 1
if 'usb' in camera_list[camera_instance]['Id']:
    webcam_color_shift = True
else:
    webcam_color_shift = False
    
picam2 = Picamera2(camera_instance)
picam2.preview_configuration.main.size = (1280, 1280)
if webcam_color_shift:
    picam2.preview_configuration.main.format = "BGR888"
else:
    picam2.preview_configuration.main.format = "RGB888"
    
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    if webcam_color_shift:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO model on the captured frame and store the results
    results = model(frame)
    
    # Output the visual detection data, we will draw this on our camera preview window
    annotated_frame = results[0].plot()
    
    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    title = 'Camera - ' + args.model

    # Display the resulting frame
    cv2.imshow(title, annotated_frame)

    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Close all windows
cv2.destroyAllWindows()
