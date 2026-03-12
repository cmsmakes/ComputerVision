## 🚀01.Calibration.py (체크보드 기반 카메라 캘리브레이션)
### 이미지에서 체크보드 코너를 검출하고 실제 좌표와 이미지 좌표의 대응 관계를 이용하여 카메라 파라미터를 추정하는 것입니다. 이를 통해 카메라의 내부 행렬과 왜곡 계수를 계산하고 이미지의 왜곡을 보정합니다.

**전체코드**

``` python

import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
# 실제 물리적 크기를 알아야 픽셀과 실제 세계의 비율을 매칭할 수 있다.
square_size = 25.0

# 코너 정밀화 조건
# 알고리즘이 반복 연산을 멈출 기준(최대 반복 횟수 30회, 정확도 0.001)을 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
# 체크보드의 교차점 개수만큼 ( x, y, Z) 좌표를 담을 (54, 3) 크기의 0으로 채워진 배열을 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# X와 Y 좌표계에 0, 1, 2 ... 순서대로 격자 형태의 인덱스 좌표를 채워 넣음
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# 위에서 만든 인덱스에 실제 체크보드 칸의 크기(25mm)를 곱해 실제 물리적 위치 좌표로 완성
objp *= square_size

# 저장할 좌표
objpoints = []  # 3D 실제 좌표 보관용
imgpoints = []  # 2D 이미지 좌표(픽셀) 보관용

# 현재 폴더에서 "left0#.jpg"라는 이름을 가진 파일들의 경로를 리스트 형태로 가져옵니다.
images = glob.glob(r"C:\Users\COM\Desktop\CV\chapter2\calibration_images\left*.jpg")

# 이미지의 가로, 세로 해상도 정보를 나중에 저장하기 위해 빈 변수로 둡니다.
img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:                    # glob으로 찾은 이미지 파일 목록을 하나씩 꺼내어 반복합니다.
    img = cv2.imread(fname)             # 이미지 파일을 읽어와 img 변수에 저장합니다.
    if img is None:                     # 이미지를 정상적으로 읽지 못했다면 반복 건너뛰고 다음 파일로 넘어감
        continue                        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 코너 검출 함수는 흑백 이미지를 필요로 하므로 그레이스케일로 변환합니다.
    
    # 이미지 크기 저장 (나중에 calibrateCamera에 사용하기 위해 가로, 세로 형태의 튜플로 저장)
    if img_size is None:
        img_size = gray.shape[::-1]
        
    # 체크보드 이미지에서 2D 이미지 좌표(코너 위치) 검출 
    # ret에는 성공 여부(True/False), corners에는 찾은 코너의 픽셀 좌표가 들어갑니다.
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    # 코너 검출에 성공한 경우에만 좌표 추가 (실패한 이미지는 제외됨) 
    if ret == True:
        objpoints.append(objp)  # 이 이미지에 대응되는 3D 실제 좌표 배열을 objpoints 리스트에 추가합니다.
        
        # 제공된 criteria를 이용하여 코너 좌표를 더 정밀하게 조정 (선택 사항이지만 정확도 향상에 도움됨)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)      # 정밀해진 2D 코너 픽셀 좌표를 imgpoints 리스트에 추가합니다.

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 모아둔 실제 좌표(objpoints)와 이미지 좌표(imgpoints)를 이용해 렌즈의 왜곡 계수와 카메라 내부 행렬을 계산합니다
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K)                    # 모아둔 실제 좌표(objpoints)와 이미지 좌표(imgpoints)를 이용해 렌즈의 왜곡 계수와 카메라 내부 행렬을 계산합니다

print("\nDistortion Coefficients:")
print(dist)                 # 렌즈의 방사 왜곡 및 접선 왜곡 계수들을 출력합니다

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0:
    print("아무 키나 누르면 다음 이미지로 넘어갑니다.")
    
    # images 리스트에 있는 모든 파일 경로를 하나씩 반복해서 꺼내옵니다.
    for fname in images:
        test_img = cv2.imread(fname)
        
        if test_img is None:
            continue
        
        # cv2.undistort()를 사용하여 왜곡 보정된 결과 생성 
        dst = cv2.undistort(test_img, K, dist, None, K)
        
        # 원본 이미지와 왜곡 보정된 이미지를 가로로 나란히 붙여서 비교 시각화
        combined_result = np.hstack((test_img, dst))
        
        # 결과 출력
        cv2.imshow('Original vs Undistorted', combined_result)
        
        # 0은 무한 대기를 의미합니다. 아무 키보드나 누르면 다음 반복(다음 이미지)으로 넘어갑니다.
        cv2.waitKey(0) 
        
    # 모든 반복이 끝나면(마지막 이미지까지 다 보면) 띄워둔 창을 닫습니다.
    cv2.destroyAllWindows()
else:
    print("경로에 이미지가 없어 시각화를 진행할 수 없습니다.")

```

**실행 결과**
<img width="1282" height="512" alt="스크린샷 2026-03-12 154918" src="https://github.com/user-attachments/assets/3d864229-65e4-4ddf-be7f-d5fef4b87ab6" />
<img width="488" height="256" alt="스크린샷 2026-03-12 160024" src="https://github.com/user-attachments/assets/104a0a51-ead2-4b2e-9838-e0b19332b82b" />


**💡 핵심 기술 요약**

**`cv2.findChessboardCorners()`**: 이미지에서 2D 체크보드 코너 좌표를 검출합니다.

**`cv2.calibrateCamera()`**: 실제 3D 좌표와 검출된 2D 이미지 좌표를 이용하여 카메라 내부 행렬(Camera matrix)과 왜곡 계수(Distortion Coefficients)를 계산합니다.

**`cv2.undistort()`**: 계산된 파라미터를 바탕으로 원본 이미지의 왜곡을 보정합니다.

---

## 🚀02.Rotation&Transformation.py ( 이미지 Rotation & Transformation )
### 한 장의 이미지에 회전(Rotation), 크기 조절(Scaling), 그리고 평행이동(Translation)을 동시에 적용하는 기하학적 변환을 실습하는 것입니다.
**전체코드**
```python

import cv2
import numpy as np

# 1. 이미지 불러오기 (테스트할 이미지 경로를 입력해주세요)
img = cv2.imread('rose.png') # 강의 자료의 장미 이미지와 같은 테스트용 이미지

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # 이미지의 높이(h)와 너비(w) 가져오기
    h, w = img.shape[:2]

    # 2. 이미지의 중심 좌표 계산
    center = (w / 2, h / 2)

    # 3. 회전 및 크기 조절 행렬 생성
    # 중심 좌표 기준으로 +30도 회전, 크기는 0.8배
    # cv2.getRotationMatrix2D() 사용
    M = cv2.getRotationMatrix2D(center, angle=30, scale=0.8)

    # 4. 평행이동 적용
    # 회전 행렬의 마지막 열(index 2) 값을 조정하여 평행이동 반영
    # x축 방향으로 +80px
    M[0, 2] += 80
    # y축 방향으로 -40px
    M[1, 2] -= 40

    # 5. 변환 적용 (cv2.warpAffine 사용)
    # 이미지에 생성한 변환 행렬 M을 적용
    result = cv2.warpAffine(img, M, (w, h))

    # 6. 결과 시각화
    # 원본 이미지와 변환된 이미지를 나란히 출력
    cv2.imshow('Original', img)
    cv2.imshow('Rotated + Scaled + Translated', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

```

**실행 결과화면**
<table align="center">
  <tr>
  <td><img width="1190" height="824" alt="스크린샷 2026-03-12 161819" src="https://github.com/user-attachments/assets/7a1ef8ef-6576-4603-9a40-af974310d5f4" />
  </td>
  <td><img width="1190" height="824" alt="스크린샷 2026-03-12 161823" src="https://github.com/user-attachments/assets/04ea4857-7c58-4a11-bc2e-01daacc5ab70" />
  </td>
  </tr>
</table>





**💡 핵심 기술 요약**

**`cv2.getRotationMatrix2D()`**: 이미지 중심을 기준으로 특정 각도(예: +30도)로 회전하고 특정 비율(예: 0.8)로 크기를 조절하는 변환 행렬을 생성합니다.

**`M[0, 2] += 80 및 M[1, 2] -= 40`**: 앞서 만든 변환 행렬은 수학적으로 각 행의 마지막 열(인덱스 2)이 평행이동(Translation)을 담당하는 구조를 가집니다. 여기에 직접 접근하여 x축 이동량(오른쪽으로 80px)과 y축 이동량(위쪽으로 40px)을 수치적으로 더해주는 핵심 로직입니다..

**`cv2.warpAffine()`**: 최종적으로 완성된 변환 행렬을 원본 이미지에 적용하여 결과 이미지를 출력합니다.

---

## 🚀03.Depth.py ( Stereo Disparity 기반 Depth 추정 )
### 같은 장면을 왼쪽과 오른쪽 카메라에서 촬영한 두 장의 이미지(Stereo images)를 이용해 물체의 깊이(Depth)를 추정하는 것입니다. 두 이미지에서 물체의 위치 차이(Disparity)를 계산하여 물체가 카메라에서 얼마나 떨어져 있는지 거리를 구합니다

**전체코드**
```python

import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
# 연산 결과물(시각화된 이미지 등)을 저장할 폴더를 안전하게 생성합니다.
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("left.png")
right_color = cv2.imread("right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 카메라 파라미터
# 3D 공간의 실제 거리를 구하기 위해 필요한 물리적 수치들입니다.
f = 700.0   # 초점 거리(focal length): 카메라 렌즈와 센서 사이의 거리 (픽셀 단위)
B = 0.12    # 베이스라인(Baseline): 왼쪽 렌즈와 오른쪽 렌즈 사이의 실제 물리적 거리

# ROI 설정
# 화면에서 거리를 측정하고 싶은 특정 물체들의 위치를 (x좌표, y좌표, 가로길이, 세로길이) 픽셀 단위로 미리 지정해 둡니다.
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
# 양쪽 이미지의 같은 물체를 찾을 때(매칭), 색상보다는 밝기 패턴(흑백)을 비교하는 것이 연산 속도가 빠르고 효율적이므로 변환합니다.
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# StereoBM 객체 생성: 이미지를 작은 블록 단위로 쪼개서 좌우 이미지의 차이를 비교하는 알고리즘입니다.
# numDisparities: 최대로 탐색할 픽셀 이동 범위, blockSize: 묶어서 비교할 블록의 크기
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# disparity(시차) 계산: 왼쪽 이미지 기준 오른쪽 이미지에서 픽셀이 얼마나 옆으로 이동했는지 계산합니다.
disp_16 = stereo.compute(left_gray, right_gray)

# 정수형 disparity 값을 실수형으로 변경하고 16으로 나누기 
# OpenCV의 compute 함수는 내부적으로 소수점 아래 정확도를 표현하기 위해 결과값에 16을 곱해(16배 스케일업) 정수형으로 반환합니다. 따라서 실제 시차 값을 얻으려면 16으로 다시 나누어야 합니다.
disparity = disp_16.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
# Disparity > 0인 픽셀만 유효한 마스크로 설정 
# 시차 값이 0이거나 음수인 경우는 프로그램이 좌우 이미지에서 같은 물체를 찾지 못해 매칭에 실패한 곳입니다. (에러 방지용)
valid_mask = disparity > 0

# depth map 초기화
depth_map = np.zeros_like(disparity, dtype=np.float32)

# 유효한 영역만 Z = f * B / d 공식 적용 
# 삼각측량 원리에 따라 실제 물체까지의 거리(Z)를 구합니다. 시차(d)가 클수록 거리는 가깝습니다.
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # 전체 이미지 중 설정해둔 특정 물체(ROI)의 영역만 슬라이싱하여 잘라냅니다.
    roi_mask = valid_mask[y:y+h, x:x+w]
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    # ROI 내 유효한 픽셀이 있는 경우만 평균 계산
    # 에러(매칭 실패)로 인해 계산되지 않은 픽셀을 제외하고, 정상적으로 거리가 계산된 픽셀들만의 평균을 구합니다.
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
# 파이썬의 min/max 함수를 이용해 평균 거리가 가장 작은(가까운) 물체와 가장 큰(먼) 물체를 자동으로 찾습니다.
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
# 시각화 데이터의 통계를 낼 때 오류 값(<=0)이 영향을 주지 못하도록 결측치(NaN, Not a Number)로 바꿉니다.
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 극단적인 노이즈 값이 전체 색상을 망치는 것을 막기 위해 하위 5%와 상위 95%의 값을 화면에 보여줄 최소/최대 기준으로 삼습니다.
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

# disparity 값을 0.0 ~ 1.0 사이의 비율 값으로 정규화(Normalization)합니다. (최소값은 0, 최대값은 1이 됨)
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

# 화면에 색상으로 표시하기 위해 0~1 값을 0~255 사이의 8비트 이미지(uint8)로 스케일링합니다.
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# 0~255의 흑백 이미지에 온도 분포처럼 색상(JET 맵: 낮을수록 파랑, 높을수록 빨강)을 입힙니다.
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
    # JET 컬러맵은 값이 클수록 빨간색입니다. 하지만 우리는 '가까운 것(작은 Depth)'을 빨간색으로 칠하고 싶으므로 값을 1.0에서 빼서 뒤집어줍니다.
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

# 원본 이미지 위에 측정 대상 물체들이 어디에 있는지 시각적으로 확인하기 위해 네모 박스와 이름을 그립니다.
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
# 결과 이미지들을 위에서 만든 output_dir 폴더에 저장합니다.
cv2.imwrite(str(output_dir / "disparity.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth.png"), depth_color)
cv2.imwrite(str(output_dir / "roi_left.png"), left_vis)

# -----------------------------
# 9. 출력
# -----------------------------
# 결과 이미지들을 새 창을 띄워 화면에 보여줍니다.
cv2.imshow("Original Left (ROI)", left_vis)
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map", depth_color)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

**실행 결과화면**
<table align="center">
  <tr>
  <td><img width="452" height="407" alt="스크린샷 2026-03-12 161839" src="https://github.com/user-attachments/assets/77db24c9-552b-416d-aebe-07050ea347da" />
  </td>
  <td><img width="452" height="407" alt="스크린샷 2026-03-12 161842" src="https://github.com/user-attachments/assets/de563212-ade0-469e-9642-608e5fc842e1" />
  </td>
    <td><img width="452" height="407" alt="스크린샷 2026-03-12 161845" src="https://github.com/user-attachments/assets/949e3645-1299-4b4f-9dd6-b6fc9f8bf2ee" />
  </td>
  </tr>
</table>






**💡 핵심 기술 요약**

**`cv2.StereoBM_create()`**: 그레이스케일로 변환된 두 이미지를 입력받아 Disparity map을 계산합니다.

**`cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`**: 컬러 이미지를 흑백(그레이스케일)으로 변환하여 블록 매칭 연산의 속도와 효율을 높입니다.

**`stereo.compute(left_gray, right_gray)`**: 변환된 양쪽 흑백 이미지를제로 비교하여 각 픽셀의 위치 차이를 담은 16배 스케일의 정수형 Disparity 맵을 추출합니다.

**`disp_16.astype(np.float32) / 16.0`**: compute 함수가 효율을 위해 반환한 정수형 데이터를 실제 거리 계산 수학 공식에 넣을 수 있도록 실수형으로 변환하고 16으로 나누어 원래 수치로 복원합니다.

**`valid_mask = disparity > 0`**: 동일한 물체를 찾지 못해 시차 값이 0이나 음수로 나온 에러 픽셀들을 걸러내어, 이후 연산에서 '0으로 나누기' 오류가 발생하지 않도록 방어막(마스크)을 씌웁니다.

**`depth_map[valid_mask] = (f * B) / disparity[valid_mask]`**: 유효한 픽셀 영역에 카메라 초점 거리(f)와 베이스라인(B)을 곱하고 시차(d)로 나누는 공식($Z=\frac{fB}{d}$)을 적용하여 실제 물체까지의 3D 깊이(Depth)를 도출합니다.

**`np.percentile(depth_valid, 5)`**: 계산된 깊이 데이터 중 비정상적으로 튀는 노이즈 값이 전체 화면의 색상 기준을 망치지 않도록, 상위 및 하위 5% 선에서 최댓값과 최솟값을 자르는(Cut-off) 기준점을 잡습니다.

**`cv2.applyColorMap(image, cv2.COLORMAP_JET)`**: 0~255 비율로 정규화된 밋밋한 흑백 화면에, 온도계처럼 수치에 따라 색이 변하는 컬러맵(낮으면 파랑, 높으면 빨강)을 입혀 눈으로 거리를 쉽게 구분하도록 만듭니다.
