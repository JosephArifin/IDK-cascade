import time

import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, models, transforms
from torchvision.models.resnet import resnet18

from config import *
from ImageNetDataset import ImageNetDataset

avg_exe_time = {
    "18": 26.23242682698583,
    "34": 71.03540536481233,
    "50": 85.249458963699,
    "101": 169.3826787023063,
    "152": 245.49235397929843,
}

# Calculated from below
confidence_thresholds = {
    "18": 0.8422207832336426,
    "34": 0.8662796020507812,
    "50": 0.842467725276947,
    "101": 0.8936317563056946,
    "152": 0.8771076202392578,
}


dataset = ImageNetDataset(data_dir, preprocess)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

target_precision = 0.95
for layers, model in models_dict.items():
    total_time = 0.0
    probs: list[tuple] = []
    num_correct = 0
    for batch_index, batch in enumerate(dataloader):
        batch = batch.to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(batch)
        end_time = time.perf_counter()
        total_time += end_time - start_time

        batch_probs = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_catid = torch.max(batch_probs, dim=0)

        is_correct = False
        if (
            "00" + "0" * (3 - len(str(top_catid.item()))) + str(top_catid.item())
            == dataset.paths[batch_index][1]
        ):
            is_correct = True
            num_correct += 1
        probs.append((float(top_prob.item()), is_correct))

    avg_exe_time[layers] = (total_time / len(dataset)) * 1000

    total = len(probs)
    probs.sort(key=lambda e: e[0])
    for i, (prob, is_correct) in enumerate(probs):
        curr_precision = num_correct / total
        if curr_precision >= target_precision:
            confidence_thresholds[layers] = prob
            break
        total -= 1
        if is_correct:
            num_correct -= 1
print(f"Avg exe times: {avg_exe_time}")
print(f"Confidence thresholds: {confidence_thresholds}")
