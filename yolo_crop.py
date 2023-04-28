import cv2

import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import json
import os

yolo_model = YOLO("yolov8n.pt")


def process_frame(frame):
    try:
        frame = Image.fromarray(np.uint8(frame)).convert('RGB')
        with torch.no_grad():
            pred = yolo_model(frame, iou=0.45, conf=0.75, classes=[0])[0]

        def _box_choise(pred):
            '''Функция выбора нужного бокса с актером
            boxes - список найденных актеров YOLO
            boxes[0], boxes[1], boxes[2] etc. -- отдельный актер относительно которого делается кроп'''
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
                return {'x1': crop_x1, 'y1': crop_y1, 'x2': crop_x2, 'y2': crop_y2}
        else:
            return {"x1": '', "y1": '', "x2": '', "y2": ''}
    except TypeError:
        return {"x1": '', "y1": '', "x2": '', "y2": ''}


def process_video(input_video_path, output_video_path):
    input_video = cv2.VideoCapture(input_video_path)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    output_list = []
    count = 0

    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while input_video.isOpened():
        ret, frame = input_video.read()
        crop_xyxy = process_frame(frame)
        output_list.append({'frame_number': count, 'crop_coordinates': crop_xyxy})
        count += 1
        if not ret:
            count = 0
            break

    input_video.release()
    output_video.release()
    return output_list


input_video_path = rf''
output_json_path = rf''

final_output = process_video(input_video_path, output_json_path)

with open(rf'{output_json_path}\{str(os.path.basename(input_video_path))}.json', 'w') as f:
    json.dump(final_output, f, indent=2)

