import cv2

import torch
from ultralytics import YOLO
from PIL import Image


yolo_model = YOLO("yolov8n.pt")


def process_frame(frame):
        with torch.no_grad():
            pred = yolo_model(frame, iou=0.45, conf=0.75, classes=[0])[0]

        def _box_choise(pred):
            'Функция выбора нужного бокса с актером'
            boxes = pred.boxes
            if len(boxes) != 0:
                return boxes[0]
            else:
                return None

        actor_box = _box_choise(pred)
        if actor_box is not None and len(actor_box.xywh) != 0:
            for xywh in actor_box[0].xywh:
                x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
                crop_x1 = int(x - (9 * w / 16) / 2)
                crop_x2 = int(x + (9 * w / 16) / 2)
                crop_y1 = int(y - (frame.size[1]) / 2)
                crop_y2 = int(y + (frame.size[1]) / 2)
                person_crop = frame.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                return  person_crop.save("test.jpg")


input_image_path = r''

input_frame = Image.open(input_image_path)
output_frame = process_frame(input_frame)


