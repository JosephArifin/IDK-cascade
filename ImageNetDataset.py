import os

from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    """
    Image Net Dataset V2
    10000 total images
    """

    def __init__(self, data_dir, transform) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.paths: list[str] = []

        for c in range(1000):
            class_dir = os.path.join(data_dir, str(c))
            image_dirs = [
                image_dir
                for image_dir in os.listdir(class_dir)
                if image_dir.endswith(".jpeg")
            ]
            self.paths.extend(os.path.join(class_dir, p) for p in image_dirs)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        input_image = Image.open(self.paths[idx]).convert("RGB")
        input_tensor = self.transform(input_image)
        return input_tensor
