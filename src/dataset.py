import numpy as np
import SimpleITK as sitk
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path

class BraTS21(Dataset):
    def __init__(self, images: Path, indices, x_transforms=None, y_transforms=None):
        self.images = images
        self.indices = indices
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        # load the image
        dir_path = list(self.images.glob('BraTS2021*'))[self.indices[idx]]
        x = torch.from_numpy(np.load(dir_path / 'X.npy'))
        y_wt = torch.from_numpy(np.load(dir_path / 'y_wt.npy'))
        y_tc = torch.from_numpy(np.load(dir_path / 'y_tc.npy'))
        y_et = torch.from_numpy(np.load(dir_path / 'y_et.npy'))

        # augmentations
        if self.x_transforms:
            x = self.x_transforms(x)
        if self.y_transforms:
            raise NotImplemented
        return x, (y_wt, y_tc, y_et)
