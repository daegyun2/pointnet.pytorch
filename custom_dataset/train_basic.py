# custom_dataset/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pointnet_regression import PointNetRegression
from dataset import PointCloudRegressionDataset
import os
import numpy as np

# 하이퍼파라미터
batch_size = 32
epochs = 100
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"현재 사용 중인 device: {device}")

# 데이터셋 & 데이터로더
dataset = PointCloudRegressionDataset(root_dir='../data/')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 모델, 손실 함수, 옵티마이저
model = PointNetRegression().to(device)
criterion_position = nn.MSELoss()
criterion_hand = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Early Stopping
best_loss = float('inf')
no_improve = 0
patience = 10

# Checkpoint 저장 디렉토리
os.makedirs('checkpoints', exist_ok=True)

# 학습 루프
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_pos_loss = 0.0
    epoch_hand_loss = 0.0
    
    hand_correct = 0
    hand_total = 0
    
    for points, body_center, hand_label in dataloader:
        points = points.to(device)
        body_center = body_center.to(device)
        hand_label = hand_label.to(device)
        
        # Transpose: (batch, N, 3) -> (batch, 3, N)
        points = points.transpose(2, 1)
        
        optimizer.zero_grad()
        
        # Forward
        pred_position, hand_logits = model(points)
        
        # Loss 계산
        loss_pos = criterion_position(pred_position, body_center)
        loss_hand = criterion_hand(hand_logits, hand_label)
        
        # Total loss (위치 loss + 0.5 * 손 loss)
        loss = loss_pos + 0.5 * loss_hand
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 통계
        epoch_loss += loss.item()
        epoch_pos_loss += loss_pos.item()
        epoch_hand_loss += loss_hand.item()
        
        # 손 분류 정확도
        _, predicted = torch.max(hand_logits, 1)
        hand_total += hand_label.size(0)
        hand_correct += (predicted == hand_label).sum().item()
    
    # 평균 계산
    avg_loss = epoch_loss / len(dataloader)
    avg_pos_loss = epoch_pos_loss / len(dataloader)
    avg_hand_loss = epoch_hand_loss / len(dataloader)
    hand_acc = 100.0 * hand_correct / hand_total
    
    # Learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # 출력
    print(f"[Epoch {epoch+1}/{epochs}] "
          f"Loss: {avg_loss:.6f} | "
          f"Pos: {avg_pos_loss:.6f} | "
          f"Hand: {avg_hand_loss:.6f} | "
          f"Hand Acc: {hand_acc:.2f}% | "
          f"LR: {current_lr:.6f}")
    
    # Scheduler 업데이트
    scheduler.step(avg_loss)
    
    # Early Stopping & Best Model 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        print(f"  ✓ New best model saved! (Loss: {best_loss:.6f})")
    else:
        no_improve += 1
        if no_improve < patience:
            print(f"  → No improvement for {no_improve}/{patience} epochs")
    
    # Early Stopping 체크
    if no_improve >= patience:
        print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
        break
    
    # 주기적 체크포인트 저장
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')
        print(f"  💾 Checkpoint saved: model_epoch_{epoch+1}.pth")

# 최종 모델 저장
torch.save(model.state_dict(), 'checkpoints/model_final.pth')
print("\n학습 완료! 최종 모델이 'checkpoints/model_final.pth'로 저장되었습니다.")
print(f"Best model: checkpoints/best_model.pth (Loss: {best_loss:.6f})")

# # custom_dataset/train.py
# import torch
# from dataset import PointCloudRegressionDataset
# from pointnet_regression import PointNetRegression
# from torch.utils.data import DataLoader
# import os

# batch_size = 32
# epochs = 100
# learning_rate = 0.0001
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"현재 사용 중인 device: {device}")

# dataset = PointCloudRegressionDataset(root_dir='../data/')
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model = PointNetRegression().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = torch.nn.MSELoss()

# os.makedirs('checkpoints', exist_ok=True)

# model.train()
# for epoch in range(epochs):
#     epoch_loss = 0.0
#     for points, gt in dataloader:
#         points, gt = points.to(device), gt.to(device)
#         points = points.transpose(2, 1)

#         optimizer.zero_grad()
#         output = model(points)
#         loss = criterion(output, gt)
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / len(dataloader)
#     print(f"[Epoch {epoch+1}/{epochs}] Avg Loss: {avg_loss:.6f}")

#     if (epoch + 1) % 10 == 0:
#         torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')

# torch.save(model.state_dict(), 'checkpoints/model_final.pth')
# print("Training finished!")
