import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset

# Dataset of RSNA ICH
class IchDataset(Dataset):
    def __init__(self, datapath, mode, transform=None, root="ICH"):
        self.datapath = datapath
        self.root = root
        self.mode = mode
        self.transform = transform

        assert self.mode in ["train", "test"]
        csv_file = os.path.join(f"data/{self.root}", self.mode+".csv")
        self.file = pd.read_csv(csv_file)

        self.image_list = self.file["id"].values
        self.targets = self.file["class"].values

        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 5, self.n_classes


    def __getitem__(self, index: int):
        image_id, target = self.image_list[index], self.targets[index]
        image = self.read_image(image_id)

        if self.transform is not None:
            image = self.transform(image)
          
        return {"image": image,
                "target": target,
                "index": index,
                "image_id": image_id}


    def __len__(self):
        return len(self.targets)


    def read_image(self, image_id):
        image_path = os.path.join(f"data/{self.root}", self.mode, self.datapath, image_id+".png")
        image = Image.open(image_path).convert("RGB")
        return image


# Dataset of ISIC 2019
class ISIC2019Dataset(Dataset):
    def __init__(self, datapath, mode, transform=None, root="ISIC2019"):
        self.datapath = datapath
        self.root = root
        self.mode = mode
        self.transform = transform

        assert self.mode in ["train", "test"]
        csv_file = os.path.join(f"data/{self.root}", self.mode+".csv")
        self.file = pd.read_csv(csv_file)

        self.image_list = self.file["id"].values
        self.targets = self.file["class"].values

        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 8, self.n_classes


    def __getitem__(self, index: int):
        image_id, target = self.image_list[index], self.targets[index]
        image = self.read_image(image_id)

        if self.transform is not None:
            image = self.transform(image)
          
        return {"image": image,
                "target": target,
                "index": index,
                "image_id": image_id}


    def __len__(self):
        return len(self.targets)


    def read_image(self, image_id):
        image_path = os.path.join(f"data/{self.root}", self.mode, self.datapath, image_id+".jpg")
        image = Image.open(image_path).convert("RGB")
        return image
    
