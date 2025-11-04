# custom_dataset/pointnet_regression.py
import torch.nn as nn
from pointnet.model import PointNetfeat

class PointNetRegression(nn.Module):
    """Multi-task: 몸통 중심 회귀 + 손 분류 (3-class)"""
    def __init__(self):
        super(PointNetRegression, self).__init__()
        
        # Feature extraction (공유)
        self.feat = PointNetfeat(global_feat=True)
        
        # Shared layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
        # Task 1: 몸통 중심 회귀 (x, y, z)
        self.fc_position = nn.Linear(256, 3)
        
        # Task 2: 손 분류 (3-class: 왼손/오른손/안잡음)
        self.fc_hand = nn.Linear(256, 3)
    
    def forward(self, x):
        # Feature extraction
        x, trans, trans_feat = self.feat(x)
        
        # Shared layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.dropout(self.fc2(x)))
        
        # Task outputs
        position = self.fc_position(x)  # (batch, 3)
        hand_logits = self.fc_hand(x)   # (batch, 3)
        
        return position, hand_logits

# # custom_dataset/pointnet_regression.py
# import torch.nn as nn
# from pointnet.model import PointNetfeat

# class PointNetRegression(nn.Module):
#     def __init__(self):
#         super(PointNetRegression, self).__init__()
#         self.feat = PointNetfeat(global_feat=True)
#         self.fc_layers = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 3)
#         )

#     def forward(self, x):
#         x, _, _ = self.feat(x)
#         return self.fc_layers(x)
