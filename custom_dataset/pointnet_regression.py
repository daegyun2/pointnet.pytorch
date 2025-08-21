# custom_dataset/pointnet_regression.py
import torch.nn as nn
from pointnet.model import PointNetfeat

class PointNetRegression(nn.Module):
    def __init__(self):
        super(PointNetRegression, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x, _, _ = self.feat(x)
        return self.fc_layers(x)
