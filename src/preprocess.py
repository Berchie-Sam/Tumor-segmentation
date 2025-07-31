import cv2
import numpy as np
import torch
from torchvision import transforms

def preprocess(input):
    if isinstance(input, str):
        image = cv2.imread(input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = input
        if image.shape[-1] == 3:  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    image = cv2.resize(image, (256, 256))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensor_img = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor_img
