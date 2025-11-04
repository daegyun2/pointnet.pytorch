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
        print(f"Dataset loaded: {len(self.input_files)} samples")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # 포인트클라우드 로드
        points = np.loadtxt(self.input_files[idx]).astype(np.float32)
        
        # GT 로드
        with open(self.gt_files[idx], 'r') as f:
            lines = [line for line in f if not line.startswith('#') and line.strip()]
            body_center = np.array(list(map(float, lines[0].split())), dtype=np.float32)
            hand_label = int(lines[1].strip())  # 0, 1, 2
        
        # 포인트 리샘플링
        if points.shape[0] >= self.npoints:
            indices = np.random.choice(points.shape[0], self.npoints, replace=False)
        else:
            indices = np.random.choice(points.shape[0], self.npoints, replace=True)
        
        points = points[indices]
        
        # 포인트만 정규화 (GT는 정규화 안 함!)
        points -= np.mean(points, axis=0)
        
        return (torch.from_numpy(points), 
                torch.from_numpy(body_center), 
                torch.tensor(hand_label, dtype=torch.long))


# # custom_dataset/dataset.py
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import glob
# import os

# class PointCloudRegressionDataset(Dataset):
#     def __init__(self, root_dir='../data/', npoints=1024):
#         self.input_files = sorted(glob.glob(os.path.join(root_dir, '*input.txt')))
#         self.gt_files = sorted(glob.glob(os.path.join(root_dir, '*gt.txt')))
#         self.npoints = npoints

#     def __len__(self):
#         return len(self.input_files)

#     def __getitem__(self, idx):
#         points = np.loadtxt(self.input_files[idx]).astype(np.float32)
#         gt = np.loadtxt(self.gt_files[idx]).astype(np.float32)

#         if points.shape[0] >= self.npoints:
#             indices = np.random.choice(points.shape[0], self.npoints, replace=False)
#         else:
#             indices = np.random.choice(points.shape[0], self.npoints, replace=True)

#         points = points[indices]
#         points -= np.mean(points, axis=0)  # normalization
#         return torch.from_numpy(points), torch.from_numpy(gt)
