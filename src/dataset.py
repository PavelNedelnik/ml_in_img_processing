import numpy as np
import SimpleITK as sitk
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path

class BraTS21(Dataset):
    def __init__(self, images: Path, indices, image_size, x_transforms=None, y_transforms=None):
        self.images = images
        self.indices = indices
        self.image_size = image_size
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms


    def __len__(self):
        return len(self.indices)
    

    def crop_images(self, x, ys, x_start, y_start, z_start):
        return [
            x[
                :,
                x_start:x_start + self.image_size[0],
                y_start:y_start + self.image_size[1],
                z_start:z_start + self.image_size[2],
        ]] + [
            y[
                x_start:x_start + self.image_size[0],
                y_start:y_start + self.image_size[1],
                z_start:z_start + self.image_size[2],
            ]
            for y in ys
        ]
        

    def __getitem__(self, idx: int):
        # load the image
        dir_path = list(self.images.glob('BraTS2021*'))[self.indices[idx]]
        x = torch.from_numpy(np.load(dir_path / 'X.npy'))
        y_wt = torch.from_numpy(np.load(dir_path / 'y_wt.npy'))
        y_tc = torch.from_numpy(np.load(dir_path / 'y_tc.npy'))
        y_et = torch.from_numpy(np.load(dir_path / 'y_et.npy'))

        # randomly crop the image
        x_start = (x.shape[1] - self.image_size[0] - 1) // 2 # random.randint(0, x.shape[1] - self.image_size[0] - 1)
        y_start = (x.shape[2] - self.image_size[1] - 1) // 2 # random.randint(0, x.shape[2] - self.image_size[1] - 1)
        z_start = (x.shape[3] - self.image_size[2] - 1) // 2 # random.randint(0, x.shape[3] - self.image_size[2] - 1)

        x, y_wt, y_tc, y_et = self.crop_images(x, [y_wt, y_tc, y_et], x_start, y_start, z_start)

        # augmentations
        if self.x_transforms:
            x = self.x_transforms(x)
        if self.y_transforms:
            raise NotImplemented
        return x, (y_wt, y_tc, y_et)
