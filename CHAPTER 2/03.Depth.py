import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("left.png")
right_color = cv2.imread("right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# 카메라 파라미터
f = 700.0
B = 0.12

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# StereoBM 객체 생성
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
# disparity 계산
disp_16 = stereo.compute(left_gray, right_gray)

# 정수형 disparity 값을 실수형으로 변경하고 16으로 나누기 
disparity = disp_16.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
# Disparity > 0인 픽셀만 유효한 마스크로 설정 
valid_mask = disparity > 0

# depth map 초기화
depth_map = np.zeros_like(disparity, dtype=np.float32)

# 유효한 영역만 Z = f * B / d 공식 적용 [cite: 115, 125]
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # ROI 영역에 해당하는 마스크
    roi_mask = valid_mask[y:y+h, x:x+w]
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    # ROI 내 유효한 픽셀이 있는 경우만 평균 계산
    if np.any(roi_mask):
        avg_disp = np.mean(roi_disp[roi_mask])
        avg_depth = np.mean(roi_depth[roi_mask])
    else:
        avg_disp = 0
        avg_depth = 0
        
    results[name] = {"avg_disp": avg_disp, "avg_depth": avg_depth}
# -----------------------------
# 4. 결과 출력
# -----------------------------
print(f"{'ROI Name':<15} | {'Avg Disparity':<15} | {'Avg Depth':<15}")
print("-" * 50)
for name, res in results.items():
    print(f"{name:<15} | {res['avg_disp']:<15.4f} | {res['avg_depth']:<15.4f}")

# 가장 가까운 ROI, 가장 먼 ROI 분석 (Depth 기준) 
closest_roi = min(results, key=lambda k: results[k]['avg_depth'])
farthest_roi = max(results, key=lambda k: results[k]['avg_depth'])

print("-" * 50)
print(f"가장 가까운 ROI: {closest_roi} (Depth: {results[closest_roi]['avg_depth']:.4f})")
print(f"가장 먼 ROI: {farthest_roi} (Depth: {results[farthest_roi]['avg_depth']:.4f})")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "disparity.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth.png"), depth_color)
cv2.imwrite(str(output_dir / "roi_left.png"), left_vis)
# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Original Left (ROI)", left_vis)
cv2.imshow("Disparity Map", disparity_color)

cv2.waitKey(0)
cv2.destroyAllWindows()