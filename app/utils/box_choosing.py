import itertools
from typing import List, Any, Dict, Optional

import numpy as np
from torch import tensor
from torchvision.models import ResNet

from utils.classifier import classify_actor


def crop(y1: float, y2: float, w: float, h: float) -> float:
    max_w = h * 9 / 16
    cur_w = y2 - y1
    crop_cord = y2
    if cur_w < max_w:
        crop_cord = crop_cord + abs(max_w - cur_w) // 2
    elif cur_w > max_w:
        crop_cord = crop_cord - abs(max_w - cur_w) // 2

    if crop_cord > w:
        crop_cord = w

    return crop_cord - max_w


def prioritise(ids: List[int], actors_priority: List[int]):
    index_dict = {}
    for i in range(len(actors_priority)):
        index_dict[actors_priority[i]] = i

    min_index = len(actors_priority)
    result = None

    for i in range(len(ids)):
        if ids[i] in index_dict:
            if index_dict[ids[i]] < min_index:
                min_index = index_dict[ids[i]]
                result = ids[i]

    return result


def is_speaking(boxes: List[tensor], video_metadata: Dict[str, Any]) -> Optional[tensor]:
    if not isinstance(video_metadata, list):
        video_metadata = [video_metadata]
    for meta, box in itertools.product(video_metadata, boxes):
        xyxy = box[0].xyxy[0]
        x1, y1, x2, y2 = xyxy[0], float(xyxy[1]), xyxy[2], float(xyxy[3])
        if meta["score"] > 0 and abs(meta["y1"] - y1) < 100.0 and abs(meta["y2"] - y2) < 100.0:
            return box[0]
    return None


def _box_choise(pred: tensor,
                frame: np.array,
                classifier: ResNet,
                actors_priority: List[int] = None,
                video_metadata: Dict[str, Any] = None) -> tensor:
    actors = {}

    boxes = pred.boxes

    if len(boxes) == 1 or (not actors_priority and not video_metadata):
        return boxes[0]
    elif len(boxes) == 0:
        return None

    for box in boxes:
        actor = classify_actor(frame, box[0].xyxy[0], classifier)
        actors[str(actor)] = box

    try:
        score = video_metadata[0].get("score", -1)
    except:
        score = video_metadata.get("score", -1)

    if video_metadata and score > 0:
        actor = is_speaking(boxes, video_metadata)
        if actor:
            return actor

    if actors_priority:
        actor = prioritise(list(actors.keys()), actors_priority)
        if actor:
            return actors[actor]

    return boxes[0]
