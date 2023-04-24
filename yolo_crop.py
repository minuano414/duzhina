import cv2

import torch
from ultralytics import YOLO


yolo_model = YOLO("yolov8n.pt")


def process_frame(frame):
    # Pre-process the image for YOLO

    # Perform actor detection
    with torch.no_grad():
        pred = yolo_model(frame, iou=0.45, conf=0.75, imgsz=720, classes=[0])[0]
    boxes = pred.boxes  # Boxes object for bbox outputs
    for xyxy, conf, cls in zip(boxes[1].xyxy, boxes[1].conf, boxes[1].cls):
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        person_crop = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return person_crop


# Read and process an image or video frame


# Read and process an image or video frame
input_image_path = r''
output_image_path = r''

input_frame = cv2.imread(input_image_path)
output_frame = process_frame(input_frame)
cv2.imwrite(rf'{input_image_path}\123.png', output_frame)
