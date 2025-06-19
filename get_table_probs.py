import os
import time
from decimal import Decimal

import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, models, transforms
from torchvision.models.resnet import resnet18

from config import *
from ImageNetDataset import ImageNetDataset
from precalculations import confidence_thresholds


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


dataset = ImageNetDataset(data_dir, preprocess)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

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
    prob_a = Decimal(0.0)
    for i in range(len(row)):
        if row[i] == 1:
            indexes.append(i)
    for r, s in table_probs.table.items():
        for i in indexes:
            if r[i] == 1:
                prob_a += Decimal(s["prob_s"])
                break
    stats["prob_a"] = float(prob_a)
print(table_probs.table)
