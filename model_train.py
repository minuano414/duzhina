

# In[4]:


import os
import cv2
import shutil
import pandas as pd
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.ops import nms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from ultralytics import YOLO


def actor_detection():
    # Set up dataset and data loader
    data_dir = r'.\train'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Set up the model and optimizer
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(dataset.classes))
    model = model.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for images, labels in data_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'actor_classification_model.pth')


def yolo_train():
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.train(data='coco128.yaml', epochs=40)
if __name__ == '__main__':
    actor_detection()
    yolo_train()
