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

# Calculated from code below
avg_exe_time = {
    "18": 26.23242682698583,
    "34": 71.03540536481233,
    "50": 85.249458963699,
    "101": 169.3826787023063,
    "152": 245.49235397929843,
}

dataset = ImageNetDataset.ImageNetDataset(data_dir, preprocess)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

with open("imagenet_classes.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

for layers, model in models_dict.items():
    total_time = 0.0
    total_batches = 0

    for batch in dataloader:
        batch = batch.to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(batch)
        end_time = time.perf_counter()
        total_time += end_time - start_time
        total_batches += 1

    avg_exe_time[layers] = (total_time / len(dataset)) * 1000
print(avg_exe_time)
