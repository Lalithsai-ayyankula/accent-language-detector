# datasets.py
import numpy as np
from torch.utils.data import Dataset
import torch

class TorchMFCCDataset(Dataset):
    def __init__(self, mfcc_list, labels):
        self.X = mfcc_list
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # n_mfcc x frames
        x = np.expand_dims(x, axis=0)  # 1 x n_mfcc x frames
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
