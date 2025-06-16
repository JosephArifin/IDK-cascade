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

# Calculated from get_avg_resnet_exe_times
avg_exe_time = {
    "18": 26.23242682698583,
    "34": 71.03540536481233,
    "50": 85.249458963699,
    "101": 169.3826787023063,
    "152": 245.49235397929843,
}


class TableProb:
    def __init__(self, num_classifiers) -> None:
        self.num_rows = 2**num_classifiers
        self.table: dict[tuple, list] = {(0,) * num_classifiers: []}

        for _ in range(self.num_rows - 1):
            next_row = self.flip(list(self.table.keys())[-1])
            self.table[next_row] = []

    def flip(self, curr_row):
        """Gets the next row like in table 2"""
        flipping = True
        next_row = []
        for bit in reversed(curr_row):
            if flipping:
                if bit == 0:
                    flipping = False
                next_row.append(int(not bit))
            else:
                next_row.append(int(bit))

        return tuple(reversed(next_row))


with open("imagenet_classes.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

model = resnet_18
for c in range(1000):
    class_dir = os.path.join(data_dir, str(c))
    image_dirs = os.listdir(class_dir)
    for image_dir in image_dirs:
        input_image = Image.open(image_dir)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
