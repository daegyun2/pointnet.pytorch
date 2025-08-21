# custom_dataset/train.py
import torch
from dataset import PointCloudRegressionDataset
from pointnet_regression import PointNetRegression
from torch.utils.data import DataLoader
import os

batch_size = 32
epochs = 100
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 사용 중인 device: {device}")

dataset = PointCloudRegressionDataset(root_dir='../data/')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = PointNetRegression().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

os.makedirs('checkpoints', exist_ok=True)

model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for points, gt in dataloader:
        points, gt = points.to(device), gt.to(device)
        points = points.transpose(2, 1)

        optimizer.zero_grad()
        output = model(points)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"[Epoch {epoch+1}/{epochs}] Avg Loss: {avg_loss:.6f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')

torch.save(model.state_dict(), 'checkpoints/model_final.pth')
print("Training finished!")
