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

resnet_18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet_18.eval()
resnet_18 = resnet_18.to(device)

resnet_34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
resnet_34.eval()
resnet_34 = resnet_34.to(device)

resnet_50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_50.eval()
resnet_50 = resnet_50.to(device)

resnet_101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
resnet_101.eval()
resnet_101 = resnet_101.to(device)

resnet_152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
resnet_152.eval()
resnet_152 = resnet_152.to(device)

models_ = [resnet_18, resnet_34, resnet_50, resnet_101, resnet_152]
model_layers = ["18", "34", "50", "101", "152"]

with open("imagenet_classes.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

class_ = 1
for i in range(10):
    file_names = get_file_names(class_)
    input_image = Image.open(file_names[i])

    input_tensor = preprocess(input_image)
    input_image = Image.open(file_names[i])
    input_batch = input_tensor.unsqueeze(0).to(device)
    print(f"File name(Debugging): {file_names[i]}")
    m = 0
    while m < 3:
        model = models_[m]

        with torch.no_grad():
            output = model(input_batch)

        probs = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_catid = torch.topk(probs, 1)
        print(f"Model: {model_layers[m]}")
        print(f"Class {categories[int(top_catid[0].item())]}: {top_prob[0].item():.4f}")

        if top_prob[0].item() >= 0.65:
            print("Success!\n")
            break
        if top_prob[0].item() < 0.3:
            print("Too unsure!\n")
            m = len(models_) - 1
            continue
        print("IDK\n")
        m += 1
