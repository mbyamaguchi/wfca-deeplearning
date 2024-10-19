import os
import gc

from tqdm import tqdm
from tifffile import TiffFile
import numpy as np
from pathlib import Path

# PyTorch 関連
import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    """PyTorch Dataset for image sequence of tif files."""

    def __init__(self, filepath1: str, filepath2: str, runrest: list, transform=None, normalize: bool=True) -> None:
        super().__init__()
        self.filepath1 = filepath1
        self.filepath2 = filepath2
        self.transform = transform
        self.normalize = normalize

        # Load image sequence of tif.
        self.ch1 = self._load_seq(self.filepath1)
        self.ch2 = self._load_seq(self.filepath2)
        
        # min-max normalize
        # self.data = self._min_max()

        # Convert to NumPy ndarray.
        self.ch1 = np.array(self.ch1)
        self.ch2 = np.array(self.ch2)
        
        # Calculate mean and std of the dataset.
        self.mean1, self.std1 = self._calculate_mean_std(self.ch1)
        self.mean2, self.std2 = self._calculate_mean_std(self.ch2)

        # Import run-rest info.
        self.label = runrest

        self.data = []
        for img1, img2 in tqdm(zip(self.ch1, self.ch2)):
            if normalize:
                tmp1 = (img1 - self.mean1) / self.std1
                tmp2 = (img2 - self.mean2) / self.std2
                img_2ch = np.array([tmp1, tmp2])
                del tmp1, tmp2
                gc.collect()
            else:
                img_2ch = np.array([img1, img2])
            self.data.append(img_2ch)
        
        # Convert to NumPy ndarray.
        self.data = np.array(self.data)

    def _load_seq(self, filepath):
        with TiffFile(filepath) as tif:
            seq = tif.asarray()
            return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        return out_data, out_label
    
    def _min_max(self):
        data = self.data
        min_value = data.min(axis=0)
        max_value = data.max(axis=0)

        data_norm = []

        for d in data:
            d = (d - min_value) / (max_value - min_value)
            data_norm.append(d)

        return data_norm
    
    def _calculate_mean_std(self, data):
        mean_data = data.mean(axis=0)
        std_data = data.std(axis=0)

        return mean_data, std_data