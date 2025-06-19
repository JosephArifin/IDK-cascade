import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, models, transforms
from torchvision.models.resnet import resnet18

from config import *
from dag import *
from precalculations import *

start_vertex = Vertex([])
dag = DAG(start_vertex, models_dict, avg_exe_time, confidence_thresholds, table_prob_a)
for i, vertex_layer in enumerate(dag.graph):
    print(f"Vertex Layer: {i}")
    for vert in vertex_layer:
        print(f"Cascade: { vert.cascade }")
