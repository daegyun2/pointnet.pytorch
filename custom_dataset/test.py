# custom_dataset/test.py
import torch
import numpy as np
from pointnet_regression import PointNetRegression
from dataset import PointCloudRegressionDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = PointNetRegression()
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.to(device)
model.eval()

# 테스트 데이터셋 로드
test_dataset = PointCloudRegressionDataset(root_dir='/home/jaden/pointnet_test_data/')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 통계 수집을 위한 리스트들
predictions = []
ground_truths = []
errors = []
hand_predictions = []
hand_ground_truths = []

print("테스트 진행 중...")
with torch.no_grad():
    for i, (points, body_center, hand_label) in enumerate(test_dataloader):
        points = points.to(device)
        body_center = body_center.to(device)
        hand_label = hand_label.to(device)
        
        points = points.transpose(2, 1)
        
        # 모델 예측
        predicted_center, hand_logits = model(points)
        
        # CPU로 이동하여 numpy 배열로 변환 (⭐ Z축 제거: [:2]만 사용)
        pred_np = predicted_center.cpu().numpy()[0][:2]  # X, Y만
        gt_np = body_center.cpu().numpy()[0][:2]        # X, Y만
        
        # 손 예측
        _, predicted_hand = torch.max(hand_logits, 1)
        pred_hand = predicted_hand.cpu().numpy()[0]
        gt_hand = hand_label.cpu().numpy()[0]
        
        # 2D 유클리드 거리 오차 계산
        error = np.linalg.norm(pred_np - gt_np)
        
        predictions.append(pred_np)
        ground_truths.append(gt_np)
        errors.append(error)
        hand_predictions.append(pred_hand)
        hand_ground_truths.append(gt_hand)
        
        # 처음 5개 샘플 출력
        if i < 5:
            hand_labels = ['왼손', '오른손', '안잡음']
            print(f"\n샘플 {i+1}:")
            print(f"  예측값 (X, Y): [{pred_np[0]:.4f}, {pred_np[1]:.4f}]")
            print(f"  실제값 (X, Y): [{gt_np[0]:.4f}, {gt_np[1]:.4f}]")
            print(f"  2D 오차: {error:.4f}m = {error*100:.2f}cm")
            print(f"  예측 손: {hand_labels[pred_hand]}")
            print(f"  실제 손: {hand_labels[gt_hand]} {'✓' if pred_hand == gt_hand else '✗'}")

# 통계 계산
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)
errors = np.array(errors)
hand_predictions = np.array(hand_predictions)
hand_ground_truths = np.array(hand_ground_truths)

print(f"\n=== 테스트 결과 통계 (2D 평면) ===")
print(f"총 테스트 샘플 수: {len(errors)}")
print(f"평균 오차 (MAE): {np.mean(errors):.4f}m = {np.mean(errors)*100:.2f}cm")
print(f"표준편차: {np.std(errors):.4f}m = {np.std(errors)*100:.2f}cm")
print(f"최소 오차: {np.min(errors):.4f}m = {np.min(errors)*100:.2f}cm")
print(f"최대 오차: {np.max(errors):.4f}m = {np.max(errors)*100:.2f}cm")
print(f"중간값 오차: {np.median(errors):.4f}m = {np.median(errors)*100:.2f}cm")

# 각 축별 오차 분석 (X, Y만)
axis_errors = np.abs(predictions - ground_truths)
print(f"\n=== 축별 평균 절대 오차 (2D) ===")
print(f"X축 MAE: {np.mean(axis_errors[:, 0]):.4f}m = {np.mean(axis_errors[:, 0])*100:.2f}cm")
print(f"Y축 MAE: {np.mean(axis_errors[:, 1]):.4f}m = {np.mean(axis_errors[:, 1])*100:.2f}cm")

# 손 분류 정확도
hand_correct = np.sum(hand_predictions == hand_ground_truths)
hand_accuracy = hand_correct / len(hand_ground_truths) * 100
print(f"\n=== 손 분류 정확도 (3-class) ===")
print(f"정확도: {hand_accuracy:.2f}% ({hand_correct}/{len(hand_ground_truths)})")

# 클래스별 정확도
for class_id, class_name in enumerate(['왼손', '오른손', '안잡음']):
    mask = hand_ground_truths == class_id
    if np.sum(mask) > 0:
        class_correct = np.sum((hand_predictions[mask] == hand_ground_truths[mask]))
        class_total = np.sum(mask)
        class_acc = 100.0 * class_correct / class_total
        print(f"  {class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")

# 정확도 분석 (2D 거리 기준)
thresholds = [0.05, 0.10, 0.15, 0.20]  # 5cm, 10cm, 15cm, 20cm
print(f"\n=== 정확도 분석 (2D 평면) ===")
for thresh in thresholds:
    accuracy = np.mean(errors < thresh) * 100
    count = np.sum(errors < thresh)
    print(f"오차 < {thresh*100:.0f}cm: {accuracy:.1f}% ({count}/{len(errors)})")

# 시각화
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # ==================================================
    # Figure 1: 기본 통계 (기존과 동일)
    # ==================================================
    fig1 = plt.figure(figsize=(20, 5))
    
    # 1. Error Histogram
    ax1 = plt.subplot(1, 4, 1)
    plt.hist(errors * 100, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    plt.xlabel('Error (cm)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Error Distribution (2D)', fontsize=12, fontweight='bold')
    plt.axvline(np.mean(errors)*100, color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(errors)*100:.2f}cm')
    plt.axvline(np.median(errors)*100, color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(errors)*100:.2f}cm')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 2. Predicted vs Actual (X axis)
    ax2 = plt.subplot(1, 4, 2)
    plt.scatter(ground_truths[:, 0], predictions[:, 0], alpha=0.3, s=10)
    min_val = min(ground_truths[:, 0].min(), predictions[:, 0].min())
    max_val = max(ground_truths[:, 0].max(), predictions[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    plt.xlabel('Actual (X)', fontsize=11)
    plt.ylabel('Predicted (X)', fontsize=11)
    plt.title('Predicted vs Actual (X Axis)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 3. Error per Sample
    ax3 = plt.subplot(1, 4, 3)
    plt.plot(errors * 100, alpha=0.5, linewidth=0.5)
    plt.axhline(y=np.mean(errors)*100, color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(errors)*100:.2f}cm')
    # 10cm, 20cm 임계선 추가
    plt.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='10cm threshold')
    plt.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='20cm threshold')
    plt.xlabel('Sample Number', fontsize=11)
    plt.ylabel('Error (cm)', fontsize=11)
    plt.title('Error per Sample (2D)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, min(50, np.max(errors)*100 + 5)])  # 최대 50cm까지만 표시
    
    # 4. Hand Classification Confusion Matrix
    ax4 = plt.subplot(1, 4, 4)
    confusion = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            confusion[i, j] = np.sum((hand_ground_truths == i) & (hand_predictions == j))
    
    labels = ['Left', 'Right', 'None']
    im = plt.imshow(confusion, cmap='Blues')
    plt.colorbar(im)
    plt.xticks([0, 1, 2], labels)
    plt.yticks([0, 1, 2], labels)
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.title(f'Hand Classification (3-class)\nAcc: {hand_accuracy:.1f}%', 
              fontsize=12, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            plt.text(j, i, confusion[i, j], ha='center', va='center', 
                    color='white' if confusion[i, j] > confusion.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_results_2d.png', dpi=200, bbox_inches='tight')
    print(f"\n기본 통계가 'test_results_2d.png'로 저장되었습니다.")
    
    # ==================================================
    # Figure 2: 개선된 경로 시각화 (3가지 방식)
    # ==================================================
    fig2 = plt.figure(figsize=(20, 6))
    
    # 2-1. Hexbin (밀도 히트맵)
    ax1 = plt.subplot(1, 3, 1)
    
    # Ground Truth 밀도
    hb_gt = plt.hexbin(ground_truths[:, 0], ground_truths[:, 1], 
                       gridsize=30, cmap='Greens', alpha=0.6, mincnt=1)
    plt.colorbar(hb_gt, label='GT Density')
    
    # 평균 경로 표시
    plt.plot(ground_truths[:, 0], ground_truths[:, 1], 
             'g-', linewidth=2, alpha=0.3, label='GT Path')
    
    plt.xlabel('X (m)', fontsize=11)
    plt.ylabel('Y (m)', fontsize=11)
    plt.title('Ground Truth Density (Hexbin)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2-2. 2D Histogram (오차 히트맵)
    ax2 = plt.subplot(1, 3, 2)
    
    # 오차를 색으로 표현
    scatter = plt.scatter(ground_truths[:, 0], ground_truths[:, 1], 
                         c=errors*100, cmap='RdYlGn_r', 
                         s=20, alpha=0.6, vmin=0, vmax=20)
    cbar = plt.colorbar(scatter, label='Error (cm)')
    cbar.set_label('Error (cm)', fontsize=10)
    
    plt.xlabel('X (m)', fontsize=11)
    plt.ylabel('Y (m)', fontsize=11)
    plt.title('Position Error Heatmap\n(Red=High Error, Green=Low Error)', 
              fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2-3. Sampled Comparison (샘플링해서 화살표로 표시)
    ax3 = plt.subplot(1, 3, 3)
    
    # 100개만 샘플링 (균등하게)
    n_samples = min(200, len(predictions))
    indices = np.linspace(0, len(predictions)-1, n_samples, dtype=int)
    
    # Ground Truth
    plt.scatter(ground_truths[indices, 0], ground_truths[indices, 1], 
               c='green', s=30, alpha=0.6, label='Ground Truth', zorder=3)
    
    # Predictions
    plt.scatter(predictions[indices, 0], predictions[indices, 1], 
               c='red', s=30, alpha=0.6, label='Predictions', zorder=3)
    
    # 오차 벡터 (화살표)
    for i in indices:
        error_i = errors[i] * 100  # cm
        # 오차가 10cm 이상인 것만 화살표로 표시
        if error_i > 10:
            plt.arrow(ground_truths[i, 0], ground_truths[i, 1],
                     predictions[i, 0] - ground_truths[i, 0],
                     predictions[i, 1] - ground_truths[i, 1],
                     head_width=0.02, head_length=0.03, 
                     fc='blue', ec='blue', alpha=0.4, linewidth=0.5)
    
    plt.xlabel('X (m)', fontsize=11)
    plt.ylabel('Y (m)', fontsize=11)
    plt.title(f'Sampled Comparison ({n_samples} samples)\nArrows: Error > 10cm', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('test_2d_analysis.png', dpi=200, bbox_inches='tight')
    print(f"개선된 경로 분석이 'test_2d_analysis.png'로 저장되었습니다.")
    
    # ==================================================
    # Figure 3: 통계적 분석 (새로 추가)
    # ==================================================
    fig3 = plt.figure(figsize=(20, 5))
    
    # 3-1. 오차 분포 (X vs Y)
    ax1 = plt.subplot(1, 4, 1)
    error_x = predictions[:, 0] - ground_truths[:, 0]
    error_y = predictions[:, 1] - ground_truths[:, 1]
    
    plt.scatter(error_x*100, error_y*100, alpha=0.3, s=10)
    plt.axhline(0, color='r', linestyle='--', linewidth=1)
    plt.axvline(0, color='r', linestyle='--', linewidth=1)
    
    # 원 그리기 (5cm, 10cm, 15cm, 20cm)
    for radius in [5, 10, 15, 20]:
        circle = plt.Circle((0, 0), radius, fill=False, 
                           color='gray', linestyle=':', alpha=0.5)
        ax1.add_patch(circle)
        plt.text(radius/np.sqrt(2), radius/np.sqrt(2), f'{radius}cm', 
                fontsize=8, color='gray')
    
    plt.xlabel('X Error (cm)', fontsize=11)
    plt.ylabel('Y Error (cm)', fontsize=11)
    plt.title('Error Distribution (X vs Y)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    max_err = max(abs(error_x*100).max(), abs(error_y*100).max())
    plt.xlim([-max_err, max_err])
    plt.ylim([-max_err, max_err])
    
    # 3-2. CDF (누적 분포)
    ax2 = plt.subplot(1, 4, 2)
    sorted_errors = np.sort(errors * 100)
    cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors) * 100
    
    plt.plot(sorted_errors, cdf, linewidth=2, color='steelblue')
    plt.axhline(50, color='orange', linestyle='--', alpha=0.7, label='50%')
    plt.axhline(90, color='red', linestyle='--', alpha=0.7, label='90%')
    plt.axhline(95, color='darkred', linestyle='--', alpha=0.7, label='95%')
    
    # 주요 백분위수 표시
    for percentile in [50, 90, 95]:
        value = np.percentile(errors * 100, percentile)
        plt.plot([value], [percentile], 'ro', markersize=8)
        plt.text(value + 1, percentile, f'{value:.1f}cm', fontsize=9)
    
    plt.xlabel('Error (cm)', fontsize=11)
    plt.ylabel('Cumulative Percentage (%)', fontsize=11)
    plt.title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, min(50, sorted_errors[-1])])
    
    # 3-3. 거리별 오차 분석
    ax3 = plt.subplot(1, 4, 3)
    distances = np.linalg.norm(ground_truths, axis=1)
    
    # 거리 구간별로 나누기
    bins = np.linspace(distances.min(), distances.max(), 10)
    bin_indices = np.digitize(distances, bins)
    
    bin_means = []
    bin_stds = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_means.append(np.mean(errors[mask] * 100))
            bin_stds.append(np.std(errors[mask] * 100))
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    
    plt.errorbar(bin_centers, bin_means, yerr=bin_stds, 
                fmt='o-', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Distance from Robot (m)', fontsize=11)
    plt.ylabel('Mean Error (cm)', fontsize=11)
    plt.title('Error vs Distance', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3-4. 각도별 오차 분석
    ax4 = plt.subplot(1, 4, 4, projection='polar')
    angles = np.arctan2(ground_truths[:, 1], ground_truths[:, 0])
    
    # 각도 구간별로 나누기 (36개 구간 = 10도씩)
    angle_bins = np.linspace(-np.pi, np.pi, 37)
    angle_indices = np.digitize(angles, angle_bins)
    
    angle_means = []
    angle_centers = []
    
    for i in range(1, len(angle_bins)):
        mask = angle_indices == i
        if np.sum(mask) > 10:  # 최소 10개 이상 샘플
            angle_means.append(np.mean(errors[mask] * 100))
            angle_centers.append((angle_bins[i-1] + angle_bins[i]) / 2)
    
    if len(angle_centers) > 0:
        ax4.plot(angle_centers, angle_means, 'o-', linewidth=2)
        ax4.fill(angle_centers, angle_means, alpha=0.3)
    
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_title('Error by Angle\n(Polar View)', fontsize=12, fontweight='bold', pad=20)
    ax4.set_ylabel('Error (cm)', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_statistical_analysis.png', dpi=200, bbox_inches='tight')
    print(f"통계 분석이 'test_statistical_analysis.png'로 저장되었습니다.")
    
except ImportError as e:
    print(f"\nmatplotlib 관련 오류: {e}")
    print("시각화를 건너뜁니다.")

print("\n테스트 완료!")

# # custom_dataset/test.py
# import torch
# import numpy as np
# from pointnet_regression import PointNetRegression
# from dataset import PointCloudRegressionDataset
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 모델 로드
# model = PointNetRegression()
# model.load_state_dict(torch.load('checkpoints/best_model.pth'))
# model.to(device)
# model.eval()

# # 테스트 데이터셋 로드
# # test_dataset = PointCloudRegressionDataset(root_dir='../data/')
# test_dataset = PointCloudRegressionDataset(root_dir='/home/jaden/pointnet_test_data/')
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # 통계 수집을 위한 리스트들
# predictions = []
# ground_truths = []
# errors = []
# hand_predictions = []
# hand_ground_truths = []

# print("테스트 진행 중...")
# with torch.no_grad():
#     for i, (points, body_center, hand_label) in enumerate(test_dataloader):
#         points = points.to(device)
#         body_center = body_center.to(device)
#         hand_label = hand_label.to(device)
        
#         points = points.transpose(2, 1)
        
#         # 모델 예측
#         predicted_center, hand_logits = model(points)
        
#         # CPU로 이동하여 numpy 배열로 변환
#         pred_np = predicted_center.cpu().numpy()[0]
#         gt_np = body_center.cpu().numpy()[0]
        
#         # 손 예측
#         _, predicted_hand = torch.max(hand_logits, 1)
#         pred_hand = predicted_hand.cpu().numpy()[0]
#         gt_hand = hand_label.cpu().numpy()[0]
        
#         # 유클리드 거리 오차 계산
#         error = np.linalg.norm(pred_np - gt_np)
        
#         predictions.append(pred_np)
#         ground_truths.append(gt_np)
#         errors.append(error)
#         hand_predictions.append(pred_hand)
#         hand_ground_truths.append(gt_hand)
        
#         # 처음 5개 샘플 출력
#         if i < 5:
#             hand_labels = ['왼손', '오른손', '안잡음']
#             print(f"\n샘플 {i+1}:")
#             print(f"  예측값: [{pred_np[0]:.4f}, {pred_np[1]:.4f}, {pred_np[2]:.4f}]")
#             print(f"  실제값: [{gt_np[0]:.4f}, {gt_np[1]:.4f}, {gt_np[2]:.4f}]")
#             print(f"  오차: {error:.4f}")
#             print(f"  예측 손: {hand_labels[pred_hand]}")
#             print(f"  실제 손: {hand_labels[gt_hand]} {'✓' if pred_hand == gt_hand else '✗'}")

# # 통계 계산
# predictions = np.array(predictions)
# ground_truths = np.array(ground_truths)
# errors = np.array(errors)
# hand_predictions = np.array(hand_predictions)
# hand_ground_truths = np.array(hand_ground_truths)

# print(f"\n=== 테스트 결과 통계 ===")
# print(f"총 테스트 샘플 수: {len(errors)}")
# print(f"평균 오차 (MAE): {np.mean(errors):.4f}")
# print(f"표준편차: {np.std(errors):.4f}")
# print(f"최소 오차: {np.min(errors):.4f}")
# print(f"최대 오차: {np.max(errors):.4f}")
# print(f"중간값 오차: {np.median(errors):.4f}")

# # 각 축별 오차 분석
# axis_errors = np.abs(predictions - ground_truths)
# print(f"\n=== 축별 평균 절대 오차 ===")
# print(f"X축 MAE: {np.mean(axis_errors[:, 0]):.4f}")
# print(f"Y축 MAE: {np.mean(axis_errors[:, 1]):.4f}")
# print(f"Z축 MAE: {np.mean(axis_errors[:, 2]):.4f}")

# # 손 분류 정확도
# hand_correct = np.sum(hand_predictions == hand_ground_truths)
# hand_accuracy = hand_correct / len(hand_ground_truths) * 100
# print(f"\n=== 손 분류 정확도 (3-class) ===")
# print(f"정확도: {hand_accuracy:.2f}% ({hand_correct}/{len(hand_ground_truths)})")

# # 클래스별 정확도
# for class_id, class_name in enumerate(['왼손', '오른손', '안잡음']):
#     mask = hand_ground_truths == class_id
#     if np.sum(mask) > 0:
#         class_correct = np.sum((hand_predictions[mask] == hand_ground_truths[mask]))
#         class_total = np.sum(mask)
#         class_acc = 100.0 * class_correct / class_total
#         print(f"  {class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")

# # 정확도 분석
# thresholds = [0.1, 0.2, 0.5, 1.0]
# print(f"\n=== 정확도 분석 ===")
# for thresh in thresholds:
#     accuracy = np.mean(errors < thresh) * 100
#     print(f"오차 < {thresh}: {accuracy:.1f}% ({np.sum(errors < thresh)}/{len(errors)})")

# # 시각화
# try:
#     plt.rcParams['font.family'] = 'DejaVu Sans'
#     plt.figure(figsize=(20, 5))
    
#     # 1. Error Histogram
#     plt.subplot(1, 4, 1)
#     plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
#     plt.xlabel('Error (Euclidean Distance)')
#     plt.ylabel('Frequency')
#     plt.title('Error Distribution')
#     plt.grid(True, alpha=0.3)
    
#     # 2. Predicted vs Actual (X axis)
#     plt.subplot(1, 4, 2)
#     plt.scatter(ground_truths[:, 0], predictions[:, 0], alpha=0.6)
#     plt.plot([ground_truths[:, 0].min(), ground_truths[:, 0].max()],
#              [ground_truths[:, 0].min(), ground_truths[:, 0].max()], 'r--')
#     plt.xlabel('Actual (X)')
#     plt.ylabel('Predicted (X)')
#     plt.title('Predicted vs Actual (X Axis)')
#     plt.grid(True, alpha=0.3)
    
#     # 3. Error per Sample
#     plt.subplot(1, 4, 3)
#     plt.plot(errors, alpha=0.7)
#     plt.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
#     plt.xlabel('Sample Number')
#     plt.ylabel('Error')
#     plt.title('Error per Sample')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 4. Hand Classification Confusion Matrix (3x3)
#     plt.subplot(1, 4, 4)
#     confusion = np.zeros((3, 3), dtype=int)
#     for i in range(3):
#         for j in range(3):
#             confusion[i, j] = np.sum((hand_ground_truths == i) & (hand_predictions == j))
    
#     labels = ['Left', 'Right', 'None']
    
#     im = plt.imshow(confusion, cmap='Blues')
#     plt.colorbar(im)
#     plt.xticks([0, 1, 2], labels)
#     plt.yticks([0, 1, 2], labels)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Hand Classification (3-class)\nAcc: {hand_accuracy:.1f}%')
    
#     # 숫자 표시
#     for i in range(3):
#         for j in range(3):
#             plt.text(j, i, confusion[i, j], ha='center', va='center', 
#                     color='white' if confusion[i, j] > confusion.max()/2 else 'black')
    
#     plt.tight_layout()
#     plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
#     print(f"\n시각화 결과가 'test_results.png'로 저장되었습니다.")
    
# except ImportError:
#     print("\nmatplotlib이 설치되지 않아 시각화를 건너뜁니다.")

# print("\n테스트 완료!")

# # # test.py
# # import torch
# # import numpy as np
# # from pointnet_regression import PointNetRegression

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # 모델 로드
# # model = PointNetRegression()
# # model.load_state_dict(torch.load('checkpoints/model_final.pth'))
# # model.to(device)
# # model.eval()

# # # 테스트할 포인트 클라우드 (예시 데이터 1개)
# # test_pointcloud = np.loadtxt('../data/sample_000604_ts1752123092.677_input.txt').astype(np.float32)

# # # 입력 데이터 전처리 (dataset과 같은 방식)
# # npoints = 1024
# # if test_pointcloud.shape[0] >= npoints:
# #     indices = np.random.choice(test_pointcloud.shape[0], npoints, replace=False)
# # else:
# #     indices = np.random.choice(test_pointcloud.shape[0], npoints, replace=True)

# # test_pointcloud = test_pointcloud[indices]
# # test_pointcloud -= np.mean(test_pointcloud, axis=0)
# # test_pointcloud = torch.from_numpy(test_pointcloud).unsqueeze(0).to(device)  # (1, N, 3)
# # test_pointcloud = test_pointcloud.transpose(2, 1)  # (1, 3, N)

# # # 모델 예측
# # with torch.no_grad():
# #     predicted_center = model(test_pointcloud)

# # print(f"예측된 몸통 중앙점: {predicted_center.cpu().numpy()[0]}")

