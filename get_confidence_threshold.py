import os
import time

import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, models, transforms
from torchvision.models.resnet import resnet18

import ImageNetDataset
from config import *

dataset = ImageNetDataset.ImageNetDataset(data_dir, preprocess)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

with open("imagenet_classes.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

for layers, model in models_dict.items():
    num_successes = 0
    for batch_index, batch in enumerate(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            output = model(batch)

        probs = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_catid = torch.topk(probs, 1)
        print(f"Class {categories[int(top_catid[0].item())]}: {top_prob[0].item():.4f}")

        if top_prob[0].item() >= 0.95:
            num_successes += 1
        break
    break
