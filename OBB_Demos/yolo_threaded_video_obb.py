import cv2
from ultralytics import YOLO
from queue import Queue
from threading import Thread

def detection_task(source, model, output_queue):
    """
    This function handles video capture, YOLO-OBB inference, and stores results in a queue.
    """
    video = cv2.VideoCapture(source)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        results = model.predict(frame)  # YOLO-OBB inference
        output_queue.put((source, results))
    video.release()

# Main execution
if __name__ == "__main__":
    output_queue = Queue()
    model = YOLO('/home/rpi5/Desktop/Yolo_Models/obb/yolov8n-obb.pt')  # Load your YOLO-OBB model
    mp4_video_path = '/home/rpi5/Desktop/Yolo_Demos/OBB_videos/4422128-hd_1280_720_30fps.mp4'  # Replace with your MP4 video file

    # Create and start the detection thread
    detection_thread = Thread(target=detection_task, args=(mp4_video_path, model, output_queue))
    detection_thread.start()

    # Continuously collect and process results
    while detection_thread.is_alive() or not output_queue.empty():
        if not output_queue.empty():
            source, results = output_queue.get()
            # Process and display your YOLO-OBB results here
            # For example:
            for r in results:
                # Access oriented bounding box data from 'obb' attribute
                boxes = r.obb.xywhr  #  x, y, width, height, rotation
                classes = r.obb.cls
                confidences = r.obb.conf

                for i, box in enumerate(boxes):
                    x, y, w, h, r_angle = box
                    # Draw rotated rectangles on the frame
                    # ... (OpenCV drawing code for oriented bounding boxes)

            # Display or save the processed frames
            cv2.imshow("YOLO-OBB Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    detection_thread.join()
    cv2.destroyAllWindows()
