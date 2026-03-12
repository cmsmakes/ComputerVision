import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = []
imgpoints = []

images = glob.glob("left01.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이미지 크기 저장 (나중에 calibrateCamera에 사용)
    if img_size is None:
        img_size = gray.shape[::-1]
        
    # 체크보드 이미지에서 2D 이미지 좌표(코너 위치) 검출 [cite: 42, 43]
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    # 코너 검출에 성공한 경우에만 좌표 추가 (실패한 이미지는 제외됨) 
    if ret == True:
        objpoints.append(objp)
        
        # 제공된 criteria를 이용하여 코너 좌표를 더 정밀하게 조정 (선택 사항이지만 정확도 향상에 도움됨)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0:
    # 테스트를 위해 첫 번째 이미지 불러오기
    test_img = cv2.imread(images[0])
    
    # cv2.undistort()를 사용하여 왜곡 보정된 결과 생성 
    dst = cv2.undistort(test_img, K, dist, None, K)
    
    # 원본 이미지와 왜곡 보정된 이미지를 가로로 나란히 붙여서 비교 시각화
    combined_result = np.hstack((test_img, dst))
    
    # 결과 출력
    cv2.imshow('Original vs Undistorted', combined_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("경로에 이미지가 없어 시각화를 진행할 수 없습니다.")
