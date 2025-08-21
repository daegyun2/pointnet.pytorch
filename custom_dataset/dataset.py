# custom_dataset/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

class PointCloudRegressionDataset(Dataset):
    def __init__(self, root_dir='../data/', npoints=1024):
        self.input_files = sorted(glob.glob(os.path.join(root_dir, '*input.txt')))
        self.gt_files = sorted(glob.glob(os.path.join(root_dir, '*gt.txt')))
        self.npoints = npoints

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        points = np.loadtxt(self.input_files[idx]).astype(np.float32)
        gt = np.loadtxt(self.gt_files[idx]).astype(np.float32)

        if points.shape[0] >= self.npoints:
            indices = np.random.choice(points.shape[0], self.npoints, replace=False)
        else:
            indices = np.random.choice(points.shape[0], self.npoints, replace=True)

        points = points[indices]
        points -= np.mean(points, axis=0)  # normalization
        return torch.from_numpy(points), torch.from_numpy(gt)
