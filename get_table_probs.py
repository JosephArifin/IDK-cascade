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

# Calculated from get_confidence_thrsholds
confidence_thresholds = {
    "18": 0.8422207832336426,
    "34": 0.8662796020507812,
    "50": 0.842467725276947,
    "101": 0.8936317563056946,
    "152": 0.8771076202392578,
}

# Calculated from below
calculated_table = {
    (0, 0, 0, 0, 0): {"count": 2397, "prob_s": 0.2397, "prob_a": 0.0},
    (0, 0, 0, 0, 1): {"count": 321, "prob_s": 0.0321, "prob_a": 0.6408},
    (0, 0, 0, 1, 0): {"count": 202, "prob_s": 0.0202, "prob_a": 0.6031},
    (0, 0, 0, 1, 1): {"count": 219, "prob_s": 0.0219, "prob_a": 0.6959000000000001},
    (0, 0, 1, 0, 0): {"count": 200, "prob_s": 0.02, "prob_a": 0.6106},
    (0, 0, 1, 0, 1): {"count": 163, "prob_s": 0.0163, "prob_a": 0.7017000000000001},
    (0, 0, 1, 1, 0): {"count": 112, "prob_s": 0.0112, "prob_a": 0.6843},
    (0, 0, 1, 1, 1): {"count": 335, "prob_s": 0.0335, "prob_a": 0.7304000000000002},
    (0, 1, 0, 0, 0): {"count": 135, "prob_s": 0.0135, "prob_a": 0.5412},
    (0, 1, 0, 0, 1): {"count": 63, "prob_s": 0.0063, "prob_a": 0.6849000000000001},
    (0, 1, 0, 1, 0): {"count": 47, "prob_s": 0.0047, "prob_a": 0.6622},
    (0, 1, 0, 1, 1): {"count": 98, "prob_s": 0.0098, "prob_a": 0.7222000000000002},
    (0, 1, 1, 0, 0): {"count": 60, "prob_s": 0.006, "prob_a": 0.6609},
    (0, 1, 1, 0, 1): {"count": 107, "prob_s": 0.0107, "prob_a": 0.7245000000000001},
    (0, 1, 1, 1, 0): {"count": 47, "prob_s": 0.0047, "prob_a": 0.7101000000000001},
    (0, 1, 1, 1, 1): {"count": 549, "prob_s": 0.0549, "prob_a": 0.7470000000000001},
    (1, 0, 0, 0, 0): {"count": 133, "prob_s": 0.0133, "prob_a": 0.4945},
    (1, 0, 0, 0, 1): {"count": 48, "prob_s": 0.0048, "prob_a": 0.68},
    (1, 0, 0, 1, 0): {"count": 23, "prob_s": 0.0023, "prob_a": 0.6554},
    (1, 0, 0, 1, 1): {"count": 48, "prob_s": 0.0048, "prob_a": 0.7208000000000001},
    (1, 0, 1, 0, 0): {"count": 48, "prob_s": 0.0048, "prob_a": 0.6517999999999999},
    (1, 0, 1, 0, 1): {"count": 68, "prob_s": 0.0068, "prob_a": 0.7219000000000001},
    (1, 0, 1, 1, 0): {"count": 36, "prob_s": 0.0036, "prob_a": 0.7084},
    (1, 0, 1, 1, 1): {"count": 235, "prob_s": 0.0235, "prob_a": 0.7468000000000001},
    (1, 1, 0, 0, 0): {"count": 31, "prob_s": 0.0031, "prob_a": 0.6051},
    (1, 1, 0, 0, 1): {"count": 29, "prob_s": 0.0029, "prob_a": 0.7089000000000001},
    (1, 1, 0, 1, 0): {"count": 15, "prob_s": 0.0015, "prob_a": 0.6919},
    (1, 1, 0, 1, 1): {"count": 85, "prob_s": 0.0085, "prob_a": 0.7403000000000002},
    (1, 1, 1, 0, 0): {"count": 37, "prob_s": 0.0037, "prob_a": 0.6861},
    (1, 1, 1, 0, 1): {"count": 129, "prob_s": 0.0129, "prob_a": 0.7401000000000002},
    (1, 1, 1, 1, 0): {"count": 69, "prob_s": 0.0069, "prob_a": 0.7282000000000002},
    (1, 1, 1, 1, 1): {"count": 3911, "prob_s": 0.3911, "prob_a": 0.7603000000000002},
}


class TableProb:
    def __init__(self, num_classifiers) -> None:
        self.num_rows = 2**num_classifiers
        self.table: dict[tuple, dict[str, float]] = {
            (0,) * num_classifiers: {"count": 0, "prob_s": 0.0, "prob_a": 0.0}
        }

        for _ in range(self.num_rows - 1):
            next_row = self.flip(list(self.table.keys())[-1])
            self.table[next_row] = {"count": 0, "prob_s": 0.0, "prob_a": 0.0}

    def flip(self, curr_row):
        """Gets the next row like in table 2"""
        flipping = True
        next_row: list[int] = []
        for bit in reversed(curr_row):
            if flipping:
                if bit == 0:
                    flipping = False
                next_row.append(int(not bit))
            else:
                next_row.append(int(bit))

        return tuple(reversed(next_row))


dataset = ImageNetDataset.ImageNetDataset(data_dir, preprocess)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

with open("imagenet_classes.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

table_probs = TableProb(len(models_dict))
for batch_index, batch in enumerate(dataloader):
    batch = batch.to(device)
    row: tuple = tuple()
    for i, (layers, model) in enumerate(models_dict.items()):
        with torch.no_grad():
            output = model(batch)

        batch_probs = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_catid = torch.max(batch_probs, dim=0)

        row += (0 if top_prob < confidence_thresholds[layers] else 1,)
    table_probs.table[row]["count"] += 1


# calculate prob-s
for row, stats in table_probs.table.items():
    stats["prob_s"] = stats["count"] / len(dataset)

# calculate prob-a
for row, stats in table_probs.table.items():
    indexes: list[int] = []
    prob_a = 0.0
    for i in range(len(row)):
        if row[i] == 1:
            indexes.append(i)
    for r, s in table_probs.table.items():
        for i in indexes:
            if r[i] == 1:
                prob_a += s["prob_s"]
                break
    stats["prob_a"] = prob_a
print(table_probs.table)
