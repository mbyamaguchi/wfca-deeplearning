import os

from tqdm import tqdm
from tifffile import TiffFile

# PyTorch 関連
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class WFCaTiffDataset(Dataset):
    """PyTorch Dataset に準じた TiffFile 用データセット"""
    def __init__(self, path: str, transform=None, train: bool=True) -> None:
        self.path = path
        self.transform = transform
        self.train = train

        self.files = os.listdir(self.path)
        self.data = []

        for file in tqdm(self.files):
            filename, ext = file.split('.')
            if ext == 'tif':
                with TiffFile(f"{self.path}/{file}") as tif:
                    self.append(tif.asarray())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]

        if self.transform:
            out_data = self.transform(out_data)
        
        return out_data