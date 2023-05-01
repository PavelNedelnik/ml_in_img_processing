"""
Script to extract the input files. Run it once.
"""

import tarfile
from pathlib import Path
import numpy as np

raw_data_path=Path('.').parent / 'data'
images = Path('.').parent / 'images'

with tarfile.open(raw_data_path / 'BraTS2021_Training_Data.tar') as f:
    f.extractall(images)