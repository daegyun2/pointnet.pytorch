# test.py
import torch
import numpy as np
from pointnet_regression import PointNetRegression
from dataset import PointCloudRegressionDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = PointNetRegression()
model.load_state_dict(torch.load('checkpoints/model_final.pth'))
model.to(device)
model.eval()

# 테스트 데이터셋 로드
test_dataset = PointCloudRegressionDataset(root_dir='../data/')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 통계 수집을 위한 리스트들
predictions = []
ground_truths = []
errors = []

print("테스트 진행 중...")
with torch.no_grad():
    for i, (points, gt) in enumerate(test_dataloader):
        points, gt = points.to(device), gt.to(device)
        points = points.transpose(2, 1)
        
        # 모델 예측
        predicted_center = model(points)
        
        # CPU로 이동하여 numpy 배열로 변환
        pred_np = predicted_center.cpu().numpy()[0]
        gt_np = gt.cpu().numpy()[0]
        
        # 유클리드 거리 오차 계산
        error = np.linalg.norm(pred_np - gt_np)
        
        predictions.append(pred_np)
        ground_truths.append(gt_np)
        errors.append(error)
        
        # 처음 5개 샘플 출력
        if i < 5:
            print(f"\n샘플 {i+1}:")
            print(f"  예측값: [{pred_np[0]:.4f}, {pred_np[1]:.4f}, {pred_np[2]:.4f}]")
            print(f"  실제값: [{gt_np[0]:.4f}, {gt_np[1]:.4f}, {gt_np[2]:.4f}]")
            print(f"  오차: {error:.4f}")

# 통계 계산
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)
errors = np.array(errors)

print(f"\n=== 테스트 결과 통계 ===")
print(f"총 테스트 샘플 수: {len(errors)}")
print(f"평균 오차 (MAE): {np.mean(errors):.4f}")
print(f"표준편차: {np.std(errors):.4f}")
print(f"최소 오차: {np.min(errors):.4f}")
print(f"최대 오차: {np.max(errors):.4f}")
print(f"중간값 오차: {np.median(errors):.4f}")

# 각 축별 오차 분석
axis_errors = np.abs(predictions - ground_truths)
print(f"\n=== 축별 평균 절대 오차 ===")
print(f"X축 MAE: {np.mean(axis_errors[:, 0]):.4f}")
print(f"Y축 MAE: {np.mean(axis_errors[:, 1]):.4f}")
print(f"Z축 MAE: {np.mean(axis_errors[:, 2]):.4f}")

# 정확도 분석 (특정 임계값 이하의 샘플 비율)
thresholds = [0.1, 0.2, 0.5, 1.0]
print(f"\n=== 정확도 분석 ===")
for thresh in thresholds:
    accuracy = np.mean(errors < thresh) * 100
    print(f"오차 < {thresh}: {accuracy:.1f}% ({np.sum(errors < thresh)}/{len(errors)})")

# 시각화 (matplotlib이 설치되어 있는 경우)
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(15, 5))
    
    # 1. Error Histogram
    plt.subplot(1, 3, 1)
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Error (Euclidean Distance)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # 2. Predicted vs Actual (X axis)
    plt.subplot(1, 3, 2)
    plt.scatter(ground_truths[:, 0], predictions[:, 0], alpha=0.6)
    plt.plot([ground_truths[:, 0].min(), ground_truths[:, 0].max()], 
             [ground_truths[:, 0].min(), ground_truths[:, 0].max()], 'r--')
    plt.xlabel('Actual (X)')
    plt.ylabel('Predicted (X)')
    plt.title('Predicted vs Actual (X Axis)')
    plt.grid(True, alpha=0.3)
    
    # 3. Error per Sample
    plt.subplot(1, 3, 3)
    plt.plot(errors, alpha=0.7)
    plt.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
    plt.xlabel('Sample Number')
    plt.ylabel('Error')
    plt.title('Error per Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print(f"\n시각화 결과가 'test_results.png'로 저장되었습니다.")
    
except ImportError:
    print("\nmatplotlib이 설치되지 않아 시각화를 건너뜁니다.")
    print("시각화를 원한다면: pip install matplotlib")

print("\n테스트 완료!")

# # test.py
# import torch
# import numpy as np
# from pointnet_regression import PointNetRegression

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 모델 로드
# model = PointNetRegression()
# model.load_state_dict(torch.load('checkpoints/model_final.pth'))
# model.to(device)
# model.eval()

# # 테스트할 포인트 클라우드 (예시 데이터 1개)
# test_pointcloud = np.loadtxt('../data/sample_000604_ts1752123092.677_input.txt').astype(np.float32)

# # 입력 데이터 전처리 (dataset과 같은 방식)
# npoints = 1024
# if test_pointcloud.shape[0] >= npoints:
#     indices = np.random.choice(test_pointcloud.shape[0], npoints, replace=False)
# else:
#     indices = np.random.choice(test_pointcloud.shape[0], npoints, replace=True)

# test_pointcloud = test_pointcloud[indices]
# test_pointcloud -= np.mean(test_pointcloud, axis=0)
# test_pointcloud = torch.from_numpy(test_pointcloud).unsqueeze(0).to(device)  # (1, N, 3)
# test_pointcloud = test_pointcloud.transpose(2, 1)  # (1, 3, N)

# # 모델 예측
# with torch.no_grad():
#     predicted_center = model(test_pointcloud)

# print(f"예측된 몸통 중앙점: {predicted_center.cpu().numpy()[0]}")

