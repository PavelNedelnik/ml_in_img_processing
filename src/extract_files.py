"""
Script to extract the input files. Run it once.
"""

import tarfile
import numpy as np
import SimpleITK as sitk
import torch
import os
from pathlib import Path
from torch.nn.functional import one_hot
from tqdm import tqdm

image_size = (128, 128, 128)
def crop_image(img):
    x_start = (img.shape[0] - image_size[0] - 1) // 2
    y_start = (img.shape[1] - image_size[1] - 1) // 2
    z_start = (img.shape[2] - image_size[2] - 1) // 2
    return img[
        x_start:x_start + image_size[0],
        y_start:y_start + image_size[1],
        z_start:z_start + image_size[2],
    ]

def unpack_file(file_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

raw_data_path=Path('.').parent / 'data'
images = Path('.').parent / 'images'

print('Unpacking files...')

with tarfile.open(raw_data_path / 'BraTS2021_Training_Data.tar') as f:
    f.extractall(images)

print('Done. Cleaning the data...')

for i, dir_path in tqdm(enumerate(images.glob('BraTS2021*'))):
    Xs = []
    for end in ['flair', 't1', 't1ce', 't2']:
        Xs.append(crop_image(unpack_file(dir_path / (dir_path.name + '_' + end + '.nii.gz')).astype('float32')))
    y = crop_image(unpack_file(dir_path / (dir_path.name + '_seg.nii.gz')).astype('uint8'))
    X = np.stack(Xs, axis=0)

    np.save(dir_path / 'X.npy', X)
    np.save(dir_path / 'y_wt.npy', (y > 0).astype('float32')) # whole tumor
    np.save(dir_path / 'y_tc.npy', ((y > 0) & (y < 4)).astype('float32')) # tumor core
    np.save(dir_path / 'y_et.npy', (y == 4).astype('float32')) # enhancing tumor

    for end in ['flair', 't1', 't1ce', 't2', 'seg']:
        os.remove(dir_path / (dir_path.name + '_' + end + '.nii.gz'))

print('Done. All  finished.')