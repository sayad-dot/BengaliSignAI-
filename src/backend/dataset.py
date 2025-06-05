# src/python/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SignDataset(Dataset):
    def __init__(self, processed_dir, classes_list):
        super().__init__()
        self.samples = []  # list of (npy_path, class_index)
        self.classes_list = classes_list

        for idx, cls in enumerate(classes_list):
            folder = os.path.join(processed_dir, cls)
            for fname in os.listdir(folder):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(folder, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, class_idx = self.samples[idx]
        video_arr = np.load(npy_path)  # shape (16, H, W, 3)
        video_arr = torch.from_numpy(video_arr).permute(3, 0, 1, 2)  # (3, 16, H, W)
        return video_arr, class_idx
