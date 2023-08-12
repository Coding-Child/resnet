import os

import torch
from torch.utils.data import Dataset

from PIL import Image


class PathologyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = list()
        self.img_labels = list()

        for label_dir in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_dir)

            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    if image_name.endswith('.jpg'):
                        self.img_paths.append(os.path.join(label_path, image_name))
                        self.img_labels.append(int(label_dir))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]

        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label
