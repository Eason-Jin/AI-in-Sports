from typing import Callable
import torch
from torch.utils.data import Dataset
from common import DEVICE


class SeriesDataset(Dataset):
    def __init__(self, sequences: dict, tau: int, target_offset_start: int = 1, target_offset_end: int = 1, transform : Callable = None):
        self.x, self.y, self.individual = self._create_dataset(
            sequences, tau, target_offset_start, target_offset_end)
        self.tau = tau
        self.target_offset_start = target_offset_start
        self.target_offset_end = target_offset_end
        self.transform = transform

    def _create_dataset(self, sequences: dict, tau: int, target_offset_start: int = 1, target_offset_end: int = 1):
        x, y, individual = [], [], []

        for key, sequence in sequences.items():
            for i in range(len(sequence)-tau-target_offset_end+1):
                feature = sequence[i:i+tau]
                target = sequence[i+target_offset_start:i +
                                  tau+target_offset_end]
                x.append(feature)
                y.append(target)
                individual.append(key)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32, device=DEVICE), individual

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx], self.individual[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample
