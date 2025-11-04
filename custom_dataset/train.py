# custom_dataset/train_with_validation.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pointnet_regression import PointNetRegression
from dataset import PointCloudRegressionDataset
import os
import numpy as np
import matplotlib.pyplot as plt


# ✅ 함수를 먼저 정의!
def plot_training_validation_history(history):
    """Train/Val 비교 그래프"""
    plt.figure(figsize=(20, 10))
    
    # 1. Total Loss (Train vs Val)
    plt.subplot(2, 4, 1)
    plt.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, label='Train')
    plt.plot(history['epoch'], history['val_loss'], 'r-', linewidth=2, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss (Train vs Val)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Position Loss (Train vs Val)
    plt.subplot(2, 4, 2)
    plt.plot(history['epoch'], history['train_pos_loss'], 'b-', linewidth=2, label='Train')
    plt.plot(history['epoch'], history['val_pos_loss'], 'r-', linewidth=2, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Position Loss')
    plt.title('Position Loss (Train vs Val)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Hand Loss (Train vs Val)
    plt.subplot(2, 4, 3)
    plt.plot(history['epoch'], history['train_hand_loss'], 'b-', linewidth=2, label='Train')
    plt.plot(history['epoch'], history['val_hand_loss'], 'r-', linewidth=2, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Hand Loss')
    plt.title('Hand Loss (Train vs Val)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Hand Accuracy (Train vs Val)
    plt.subplot(2, 4, 4)
    plt.plot(history['epoch'], history['train_hand_acc'], 'b-', linewidth=2, label='Train')
    plt.plot(history['epoch'], history['val_hand_acc'], 'r-', linewidth=2, label='Val')
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Hand Accuracy (Train vs Val)')
    plt.ylim([0, 105])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Learning Rate
    plt.subplot(2, 4, 5)
    plt.plot(history['epoch'], history['lr'], 'orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 6. Loss Gap (Train - Val)
    plt.subplot(2, 4, 6)
    loss_gap = np.array(history['train_loss']) - np.array(history['val_loss'])
    plt.plot(history['epoch'], loss_gap, 'purple', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Gap')
    plt.title('Loss Gap (Train - Val)')
    plt.grid(True, alpha=0.3)
    
    # 7. Accuracy Gap (Train - Val)
    plt.subplot(2, 4, 7)
    acc_gap = np.array(history['train_hand_acc']) - np.array(history['val_hand_acc'])
    plt.plot(history['epoch'], acc_gap, 'green', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Overfitting threshold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.title('Accuracy Gap (Train - Val)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Val Loss with Best Point
    plt.subplot(2, 4, 8)
    plt.plot(history['epoch'], history['val_loss'], 'r-', linewidth=2)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = np.min(history['val_loss'])
    plt.plot(best_epoch, best_val_loss, 'g*', markersize=15, label=f'Best: Epoch {best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.title('Validation Loss (Best Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_validation_history.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================
# 메인 학습 코드
# ============================================

# 하이퍼파라미터
batch_size = 256
epochs = 100
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"현재 사용 중인 device: {device}")

# Train/Val Split (80:20)
full_dataset = PointCloudRegressionDataset(root_dir='../data/')
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train: {train_size} samples, Val: {val_size} samples")

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

# 학습 기록용 리스트 (Train + Val)
history = {
    'epoch': [],
    'train_loss': [],
    'train_pos_loss': [],
    'train_hand_loss': [],
    'train_hand_acc': [],
    'val_loss': [],
    'val_pos_loss': [],
    'val_hand_loss': [],
    'val_hand_acc': [],
    'lr': []
}

# 학습 루프
for epoch in range(epochs):
    # ========== Training ==========
    model.train()
    train_loss = 0.0
    train_pos_loss = 0.0
    train_hand_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for points, body_center, hand_label in train_loader:
        points = points.to(device)
        body_center = body_center.to(device)
        hand_label = hand_label.to(device)
        
        points = points.transpose(2, 1)
        
        optimizer.zero_grad()
        
        pred_position, hand_logits = model(points)
        
        loss_pos = criterion_position(pred_position, body_center)
        loss_hand = criterion_hand(hand_logits, hand_label)
        loss = loss_pos + 0.5 * loss_hand
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_pos_loss += loss_pos.item()
        train_hand_loss += loss_hand.item()
        
        _, predicted = torch.max(hand_logits, 1)
        train_total += hand_label.size(0)
        train_correct += (predicted == hand_label).sum().item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_pos = train_pos_loss / len(train_loader)
    avg_train_hand = train_hand_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    
    # ========== Validation ==========
    model.eval()
    val_loss = 0.0
    val_pos_loss = 0.0
    val_hand_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for points, body_center, hand_label in val_loader:
            points = points.to(device)
            body_center = body_center.to(device)
            hand_label = hand_label.to(device)
            
            points = points.transpose(2, 1)
            
            pred_position, hand_logits = model(points)
            
            loss_pos = criterion_position(pred_position, body_center)
            loss_hand = criterion_hand(hand_logits, hand_label)
            loss = loss_pos + 0.5 * loss_hand
            
            val_loss += loss.item()
            val_pos_loss += loss_pos.item()
            val_hand_loss += loss_hand.item()
            
            _, predicted = torch.max(hand_logits, 1)
            val_total += hand_label.size(0)
            val_correct += (predicted == hand_label).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_pos = val_pos_loss / len(val_loader)
    avg_val_hand = val_hand_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    
    # Learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # 기록 저장
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(avg_train_loss)
    history['train_pos_loss'].append(avg_train_pos)
    history['train_hand_loss'].append(avg_train_hand)
    history['train_hand_acc'].append(train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_pos_loss'].append(avg_val_pos)
    history['val_hand_loss'].append(avg_val_hand)
    history['val_hand_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    # 출력
    print(f"[Epoch {epoch+1}/{epochs}]")
    print(f"  Train - Loss: {avg_train_loss:.6f} | Pos: {avg_train_pos:.6f} | "
          f"Hand: {avg_train_hand:.6f} | Acc: {train_acc:.2f}%")
    print(f"  Val   - Loss: {avg_val_loss:.6f} | Pos: {avg_val_pos:.6f} | "
          f"Hand: {avg_val_hand:.6f} | Acc: {val_acc:.2f}%")
    print(f"  LR: {current_lr:.6f}")
    
    # Overfitting 체크
    if train_acc - val_acc > 5.0:
        print(f"  ⚠️  Overfitting 감지! (Train Acc: {train_acc:.1f}% vs Val Acc: {val_acc:.1f}%)")
    
    # Scheduler 업데이트 (Val Loss 기준)
    scheduler.step(avg_val_loss)
    
    # Early Stopping (Val Loss 기준)
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        print(f"  ✓ New best model saved! (Val Loss: {best_loss:.6f})")
    else:
        no_improve += 1
        if no_improve < patience:
            print(f"  → No improvement for {no_improve}/{patience} epochs")
    
    # 매 10 epoch마다 그래프 업데이트
    if (epoch + 1) % 10 == 0:
        plot_training_validation_history(history)
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')
        print(f"  💾 Checkpoint saved: model_epoch_{epoch+1}.pth")
    
    # Early Stopping 체크
    if no_improve >= patience:
        print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
        break
    
    print()  # 빈 줄

# 최종 모델 저장
torch.save(model.state_dict(), 'checkpoints/model_final.pth')

# 최종 그래프 생성
plot_training_validation_history(history)

print("\n학습 완료! 최종 모델이 'checkpoints/model_final.pth'로 저장되었습니다.")
print(f"Best model: checkpoints/best_model.pth (Val Loss: {best_loss:.6f})")
print(f"학습 그래프가 'training_validation_history.png'로 저장되었습니다.")