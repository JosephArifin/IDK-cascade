import torch
from torchvision import models, transforms

data_dir = "./data/imagenetv2-top-images-format-val/"
device = "cuda" if torch.cuda.is_available() else "cpu"


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

models_dict = {
    "18": resnet_18,
    "34": resnet_34,
    "50": resnet_50,
    "101": resnet_101,
    "152": resnet_152,
}
