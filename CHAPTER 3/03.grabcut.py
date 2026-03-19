import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img = cv.imread('coffee cup.JPG')
# 이미지와 동일한 크기의 마스크(0으로 초기화)를 생성
mask = np.zeros(img.shape[:2], np.uint8)

# GrabCut 알고리즘 내부 연산에서 사용할 배경(bgd) 및 전경(fgd) 임시 모델을 초기화
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 3. 초기 사각형 영역 설정 (x, y, w, h)
# 이미지 크기에 맞춰 적절히 조정이 필요합니다.
rect = (50, 50, img.shape[1]-100, img.shape[0]-100)

# 4. cv.grabCut() 수행 ( 전경과 배경을 분리 5회 반복 )
# GC_INIT_WITH_RECT 옵션은 방금 설정한 사각형(rect) 정보를 초기값으로 쓴다는 의미
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# 5. 마스크 값을 확실한 배경 : 0 또는 애매한 배경 : 1로 변경
# cv.GC_BGD(0), cv.GC_PR_BGD(2)는 0으로, 나머지는 1로 설정
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 6. 원본 이미지에 마스크를 곱하여 배경 제거
result = img * mask2[:, :, np.newaxis]

# 7. 시각화 (원본, 마스크, 결과)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(mask2, cmap='gray')
plt.title('Mask')

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title('Foreground Extraction')

plt.show()