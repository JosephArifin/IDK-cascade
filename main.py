import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, models, transforms
from torchvision.models.resnet import resnet18

data_dir = "./data/imagenetv2-top-images-format-val/"
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_file_names(c):
    path = data_dir + str(c) + "/"
    return [path + img for img in os.listdir(path) if img[-5:] == ".jpeg"]


preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()
model = model.to(device)


for i in range(10):
    file_names = get_file_names(0)
    input_image = Image.open(file_names[0])
    print(file_names[0])

    input_tensor = preprocess(input_image)
    input_image = Image.open(file_names[0])
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    with open("imagenet_classes.txt", "r") as f:
        categories = [line.strip() for line in f.readlines()]

    probs = torch.nn.functional.softmax(output[0], dim=0)

    top_5_prob, top_5_catid = torch.topk(probs, 5)
    for j in range(5):
        print(
            f"Class {categories[int(top_5_catid[j].item())]}: {top_5_prob[j].item():.4f}"
        )
