import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1. 이미지 불러오기 [cite: 52]
img1 = cv.imread('img1.jpg') 
img2 = cv.imread('img2.jpg') 

# 2. SIFT 특징점 검출 [cite: 53]
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3. FLANN 매처 및 Ratio Test 적용 [cite: 54, 62]
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

if len(good_matches) > 10:
    # --- 데이터 준비 1: 특징점 매칭 시각화 --- [cite: 40]
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, 
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

    # --- 데이터 준비 2: 이미지 정합 (Panorama) ---
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 호모그래피 계산 [cite: 55, 60]
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 이미지 변환 및 합성 [cite: 56, 61]
    warped_img = cv.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
    result = warped_img.copy()
    result[0:h1, 0:w1] = img1
    result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    # --- 최종 출력: 하나의 Figure에 두 Plot 배치 --- [cite: 57]
    plt.figure(figsize=(12, 10)) # 전체 창 크기 설정

    # 첫 번째 칸 (위): 매칭 결과
    plt.subplot(2, 1, 1) # 2행 1열 중 1번째
    plt.imshow(img_matches_rgb)
    plt.title('Step 1: Feature Matching (Correspondences)')
    plt.axis('off')

    # 두 번째 칸 (아래): 정합 결과
    plt.subplot(2, 1, 2) # 2행 1열 중 2번째
    plt.imshow(result_rgb)
    plt.title('Step 2: Final Stitched Panorama')
    plt.axis('off')

    plt.tight_layout() # 이미지 간 간격 자동 조정
    plt.show()

else:
    print("충분한 매칭점을 찾지 못했습니다.")