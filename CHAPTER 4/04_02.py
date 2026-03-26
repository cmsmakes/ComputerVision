import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1. 이미지 불러오기 (탐색기에 있는 실제 파일명 '83'으로 수정) [cite: 34]
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color80.jpg')

# 2. SIFT 객체 생성 및 특징점/기술자 추출 [cite: 38]
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3. FLANN 매처 설정 [cite: 39, 43]
# FLANN_INDEX_KDTREE = 1 (SIFT, SURF 등 알고리즘에 적합)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100) # 탐색 횟수 설정

flann = cv.FlannBasedMatcher(index_params, search_params)

# 4. knnMatch 수행 (최근접 이웃 2개 추출) 
matches = flann.knnMatch(des1, des2, k=2)

# 5. Lowe's Ratio Test 적용 (매칭 정확도 향상) 
# 첫 번째 매칭점과 두 번째 매칭점의 거리 비율이 0.7 미만인 경우만 선택
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 6. 매칭 결과 시각화 [cite: 40]
# 색상 반전 방지를 위해 RGB 변환을 적용하여 출력합니다.
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, 
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 7. 결과 출력 (Matplotlib 사용) [cite: 41]
plt.figure(figsize=(15, 8))
# BGR을 RGB로 변환하여 정상적인 색상으로 표시
plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
plt.title('SIFT Feature Matching (FLANN + Ratio Test)')
plt.axis('off')
plt.show()