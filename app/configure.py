import torch
import torchvision
from ultralytics import YOLO
from torchvision.models import ResNet


def load_classifier(model_path: str) -> ResNet:
    model = torchvision.models.resnet34()

    model.fc = torch.nn.Linear(model.fc.in_features, 96)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_detector() -> YOLO:
    return YOLO("../models/yolov8n.pt")
