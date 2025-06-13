import torch
from torch.cpu import is_available
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, models, transforms
from torchvision.models.resnet import resnet18

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225]),
    ]
)

data_dir = "./data/imagenetv2-top-images-format-val/"
dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
data_batch = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_18 = models.resnet18(pretrained=True)
resnet_18 = resnet_18.to(device)
