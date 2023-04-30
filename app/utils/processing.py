from typing import List, Optional, Any, Dict

import cv2
import numpy as np
import torch
from torchvision.models import ResNet
from ultralytics import YOLO

from utils.box_choosing import _box_choise, crop


def process_frame(frame: np.array,
                  count: int,
                  width: float,
                  height: float,
                  detector: YOLO,
                  classifier: ResNet,
                  actors_priorty: List[int] = None, video_metadata: List[int] = None) -> Optional[float]:
    with torch.no_grad():
        pred = detector(frame, iou=0.45, conf=0.75, classes=[0])[0]
    if video_metadata:
        actor_box = _box_choise(pred, frame, classifier, actors_priorty, video_metadata[count])
    else:
        actor_box = _box_choise(pred, frame, classifier, actors_priorty)
    if actor_box:
        for xyxy in actor_box[0].xyxy:
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            return crop(y1, y2, width, height)
    else:
        return None


def process_video(input_video_path: str,
                  detector: YOLO,
                  classifier: ResNet,
                  actors_priorty: List[int] = None,
                  video_metadata: Dict[str, Any] = None):
    input_video = cv2.VideoCapture(input_video_path)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_amount = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_list = []
    count = 0

    while input_video.isOpened():
        ret, frame = input_video.read()
        crop_y = process_frame(frame=frame,
                               count=count,
                               width=width,
                               height=height,
                               detector=detector,
                               classifier=classifier,
                               actors_priorty=actors_priorty,
                               video_metadata=video_metadata)
        output_list.append({'frame_number': count, 'crop_coordinates': float(crop_y) if crop_y else None})
        count += 1
        if not ret or count == frames_amount:
            break

    input_video.release()
    return output_list
