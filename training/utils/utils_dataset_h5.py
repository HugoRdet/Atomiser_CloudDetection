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

import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist

class CloudSEN12Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None,channels=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        if channels==None:
            self.channels=[1,2,3,4,5,6,7,8,9,10,11,12]
        else:
            self.channels=channels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        #self.indices[i].item()
        s2_l1c = self.dataset.read(i).read(0)
        s2_label = self.dataset.read(i).read(1)

        with rio.open(s2_l1c) as src, rio.open(s2_label) as dst:
            image = src.read(self.channels,window=rio.windows.Window(0, 0, 512, 512))
            mask = dst.read([1],window=rio.windows.Window(0, 0, 512, 512))

        if self.transform:
            image = self.transform(image)

        return image, mask
    


class CloudSEN12DataModule(pl.LightningDataModule):
    def __init__(self, dataset,transform=None, batch_size=16, num_workers=4):
        super().__init__()
        self.dataset = dataset
        
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_path = Path("./data/sen12_train.pt")
        if train_path.exists():
            train_idx = torch.load(train_path)
        else:
            raise FileNotFoundError(f"{train_path} not found")
        
        validation_path = Path("./data/sen12_validation.pt")
        if train_path.exists():
            validation_idx = torch.load(train_path)
        else:
            raise FileNotFoundError(f"{train_path} not found")
        
        test = Path("./data/sen12_test.pt")
        if train_path.exists():
            test_idx = torch.load(train_path)
        else:
            raise FileNotFoundError(f"{train_path} not found")
        
        self.train_idx = train_idx
        self.val_idx = validation_idx
        self.test_idx = test_idx

    def setup(self, stage=None):
        self.train_dataset = CloudSEN12Subset(self.dataset, self.train_idx, transform=self.transform)
        self.val_dataset   = CloudSEN12Subset(self.dataset, self.val_idx, transform=self.transform)
        self.test_dataset  = CloudSEN12Subset(self.dataset, self.test_idx, transform=self.transform)

    def train_dataloader(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Train DataLoader created")
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Validation DataLoader created")
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Test DataLoader created")
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          pin_memory=True, persistent_workers=True)