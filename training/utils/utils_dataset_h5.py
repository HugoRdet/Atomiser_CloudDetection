import h5py
import os
import torch
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import einops as einops
from torch.utils.data import Dataset,DataLoader,Sampler
import h5py
from tqdm import tqdm
from .image_utils import*
from .utils_dataset import*
import random
from .FLAIR_2 import*
from datetime import datetime, timezone
import torch.distributed as dist
import time
import rasterio as rio

def random_flip_and_rotate(x, y):
    """
    Apply the same random flip and rotation to both x and y.

    Args:
        x (Tensor): shape [C, H, W]
        y (Tensor): shape [H, W]

    Returns:
        (x, y): transformed tensors
    """
    # Random rotation (0, 90, 180, 270 degrees)
    k = random.randint(0, 3)
    x = torch.rot90(x, k, dims=[1, 2])  # rotate along H and W
    y = torch.rot90(y, k, dims=[0, 1])

    # Random horizontal flip
    if random.random() > 0.5:
        x = torch.flip(x, dims=[2])  # flip W
        y = torch.flip(y, dims=[1])

    # Random vertical flip
    if random.random() > 0.5:
        x = torch.flip(x, dims=[1])  # flip H
        y = torch.flip(y, dims=[0])

    return x, y

class CloudSEN12Dataset(Dataset):
    def __init__(self, mode, mean_path="./data/cloudsen12/mean.pt", std_path="./data/cloudsen12/std.pt", transform=None,binary=False):
        """
        Args:
            data_dir (str): Path to directory with .pt files.
            mean_path (str): Path to mean.pt (tensor of shape [12]).
            std_path (str): Path to std.pt (tensor of shape [12]).
            transform (callable, optional): Optional transform to apply on x and y.
        """
        self.data_dir = "./data/cloudsen12/cloudsen12_"+mode
        self.file_list = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".pt")])
        self.transform=transform
        self.mean = torch.from_numpy(torch.load(mean_path, weights_only=False)).float()
        self.std = torch.from_numpy(torch.load(std_path, weights_only=False)).float()   
        self.binary=binary

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            file_path = os.path.join(self.data_dir, self.file_list[idx])
            sample = torch.load(file_path)  # [H, W, 13]

            x = sample[..., :12].float()
            y = sample[..., 12].long()

            x = (x - self.mean) / self.std
            x = x.permute(2, 0, 1)
            y = y.squeeze()
            

            if self.transform:
                x, y = self.transform(x, y)

            if self.binary:
                y[y==3]=0 #means 0 -> not cloud
                y[y==2]=1 #      1 -> cloud
            

            return x, y

        except Exception as e:
            print(f"❌ Error in __getitem__({idx}): {e}")
            raise e


import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class CloudSEN12DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="./data/cloudsen12",
        batch_size=8,
        num_workers=4,
        mean_path="./data/cloudsen12/mean.pt",
        std_path="./data/cloudsen12/std.pt",
        binary=False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean_path = mean_path
        self.std_path = std_path
        self.binary=binary

    def setup(self, stage=None):
        

        if stage == "fit" or stage is None:
            self.train_dataset = CloudSEN12Dataset(
                mode="train",
                mean_path=self.mean_path,
                std_path=self.std_path,
                transform=random_flip_and_rotate,
                binary=self.binary
            )
            self.val_dataset = CloudSEN12Dataset(
                mode="validation",
                mean_path=self.mean_path,
                std_path=self.std_path,
                transform=None,
                binary=self.binary
            )

        if stage == "test" or stage is None:
            self.test_dataset = CloudSEN12Dataset(
                mode="test",
                mean_path=self.mean_path,
                std_path=self.std_path,
                transform=None,
                binary=self.binary
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )



import os
import torch
from torch.utils.data import Dataset

class Cloud95Dataset(Dataset):
    def __init__(self, mode, data_dir="./data/95_cloud", mean_path=None, std_path=None, transform=None):
        """
        Args:
            mode (str): One of ["train", "validation", "test"]
            data_dir (str): Root directory with 95_train/, 95_validation/, 95_test/
            mean_path, std_path: Paths to mean.pt and std.pt (shape [4])
        """
        self.data_dir = os.path.join(data_dir, f"95_{mode}")
        self.file_list = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".pt")])

        self.mean = torch.load(mean_path, weights_only=False) if mean_path else None
        self.std = torch.load(std_path, weights_only=False) if std_path else None
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        sample = torch.load(file_path)  # [H, W, 5]

        x = sample[..., :4].float()  # RGBNir
        y = sample[..., 4].long()    # GT

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        x = x.permute(2, 0, 1)  # [C, H, W]
        y = y.squeeze()         # [H, W]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y



import pytorch_lightning as pl
from torch.utils.data import DataLoader

class Cloud95DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="./data/95_cloud",
        batch_size=8,
        num_workers=4,
        mean_path="./data/95_cloud/mean.pt",
        std_path="./data/95_cloud/std.pt"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean_path = mean_path
        self.std_path = std_path

    def setup(self, stage=None):


        if stage == "fit" or stage is None:
            self.train_dataset = Cloud95Dataset("train", self.data_dir, self.mean_path, self.std_path, transform=random_flip_and_rotate)
            self.val_dataset = Cloud95Dataset("validation", self.data_dir, self.mean_path, self.std_path, transform=None)

        if stage == "test" or stage is None:
            self.test_dataset = Cloud95Dataset("train", self.data_dir, self.mean_path, self.std_path, transform=None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
