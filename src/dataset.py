import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from pathlib import Path


class BraTS21(Dataset):
    @staticmethod
    def unpack_file(file: Path) -> np.array:
        return sitk.GetArrayFromImage(sitk.ReadImage(file))

    def __init__(self, images: Path):
        self.images = images
        self.X_endings = ['flair', 't1', 't1ce', 't2']
        self.y_ending = 'seg'
        self.extension = 'nii.gz'

    def __len__(self):
        return len(list(self.images.glob('*')))

    def __getitem__(self, idx: int): # -> TODO
        dir_path = list(self.images.glob('*'))[idx]
        Xs = []
        for end in self.X_endings:
            Xs.append(self.unpack_file(dir_path / (dir_path.name + '_' + end + '.' + self.extension)))

        y = self.unpack_file(dir_path / (dir_path.name + '_' + self.y_ending + '.' + self.extension))

        return Xs, y
