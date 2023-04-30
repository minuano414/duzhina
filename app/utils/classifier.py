import numpy as np
import torch
import PIL.Image as Image
from torch import tensor
from torchvision.models import ResNet
from torchvision.transforms import functional as F


def classify_actor(image: np.array, bbox: tensor, classifier: ResNet) -> int:
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    image = Image.fromarray(image)
    crop_image = image.crop((x1, y1, x2, y2))
    crop_image = F.resize(crop_image, (224, 224))
    crop_image = F.to_tensor(crop_image)
    crop_image = F.normalize(crop_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    with torch.no_grad():
        class_probs = torch.softmax(classifier(crop_image.unsqueeze(0)), dim=1)
        top_class_id = class_probs.argmax(dim=1).item()
        return top_class_id
