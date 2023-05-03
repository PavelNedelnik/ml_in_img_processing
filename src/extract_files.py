"""
Script to extract the input files. Run it once.
"""

import tarfile
import numpy as np
import SimpleITK as sitk
import torch
from pathlib import Path
from constants import *
from tqdm import tqdm

def unpack_file(file_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

def crop_image(img):
    return img[
        (img.shape[0] - IMG_SIZE[0]) // 2:(img.shape[0] - IMG_SIZE[0]) // 2 + IMG_SIZE[0],
        (img.shape[1] - IMG_SIZE[1]) // 2:(img.shape[1] - IMG_SIZE[1]) // 2 + IMG_SIZE[1],
        (img.shape[2] - IMG_SIZE[2]) // 2:(img.shape[2] - IMG_SIZE[2]) // 2 + IMG_SIZE[2],
    ]

raw_data_path=Path('.').parent / 'data'
images = Path('.').parent / 'images'

print('Unpacking files...')

"""
with tarfile.open(raw_data_path / 'BraTS2021_Training_Data.tar') as f:
    f.extractall(images)
"""

print('Done. Proprocessing...')

for dir_path in tqdm(images.glob('BraTS2021*')):
    Xs = []
    for end in ['flair', 't1', 't1ce', 't2']:
        Xs.append(crop_image(unpack_file(dir_path / (dir_path.name + '_' + end + '.nii.gz'))))
    y = crop_image(unpack_file(dir_path / (dir_path.name + '_seg.nii.gz')))
    X = np.stack(Xs, axis=3)

    torch.save(torch.from_numpy((X - X.mean()) / X.var()).float().permute(3, 0, 1, 2), dir_path / 'X.pt')
    torch.save(torch.from_numpy((y - y.mean()) / y.var()).float(), dir_path / 'y.pt')

print('Done. All  finished.')