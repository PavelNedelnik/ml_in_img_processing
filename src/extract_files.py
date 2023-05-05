"""
Script to extract the input files. Run it once.
"""

import tarfile
import numpy as np
import SimpleITK as sitk
import torch
from pathlib import Path
from torch.nn.functional import one_hot
from tqdm import tqdm


def unpack_file(file_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

raw_data_path=Path('.').parent / 'data'
images = Path('.').parent / 'images'

print('Unpacking files...')

# TODO uncomment
"""
with tarfile.open(raw_data_path / 'BraTS2021_Training_Data.tar') as f:
    f.extractall(images)
"""

print('Done. Cleaning the data...')

for i, dir_path in tqdm(enumerate(images.glob('BraTS2021*'))):
    Xs = []
    for end in ['flair', 't1', 't1ce', 't2']:
        Xs.append(unpack_file(dir_path / (dir_path.name + '_' + end + '.nii.gz')).astype('float32'))
    y = unpack_file(dir_path / (dir_path.name + '_seg.nii.gz')).astype('uint8')
    X = np.stack(Xs, axis=0)

    np.save(dir_path / 'X.npy', X)
    np.save(dir_path / 'y_wt.npy', (y > 0).astype('float32')) # whole tumor
    np.save(dir_path / 'y_tc.npy', ((y > 0) & (y < 4)).astype('float32')) # tumor core
    np.save(dir_path / 'y_et.npy', (y == 4).astype('float32')) # enhancing tumor

print('Done. All  finished.')