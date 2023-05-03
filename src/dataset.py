import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BraTS21(Dataset):
    def __init__(self, images: Path, indices, augmentations=[]):
        self.images = images
        self.indices = indices
        self.augmentations = augmentations

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int): # -> TODO
        dir_path = list(self.images.glob('BraTS2021*'))[self.indices[idx]]
        X, y = torch.load(dir_path / 'X.pt'), torch.load(dir_path / 'y.pt')
        for augment in self.augmentations:
            X = augment(X)
        return X, y
