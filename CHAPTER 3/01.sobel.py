import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. cv.imread()를 사용하여 이미지 불러오기
img = cv.imread('edgeDetectionImage.jpg') 

# 2. cv.cvtColor()를 사용하여 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 3. cv.Sobel()을 사용하여 x축과 y축 방향 에지 검출 
# 1, 0은 x축 방향 미분, ksize=3은 3x3 크기의 소벨 커널을 사용함을 의미
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
# 0, 1은 y축 방향 미분을 의미
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 4. cv.magnitude()를 사용하여 x축과 y축 에지 강도를 합산하여 전체 에지 강도 계산
magnitude = cv.magnitude(sobel_x, sobel_y)

# 5. cv.convertScaleAbs()를 사용하여 8비트 정수 형식으로 변환
sobel_magnitude = cv.convertScaleAbs(magnitude)

# 6. Matplotlib를 사용하여 시각화
# 첫 번째 칸에 원본 이미지 출력
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
# 두 번째 칸에 최종 에지 강도 이미지를 흑백(gray) 모드로 출력
plt.subplot(1, 2, 2)
plt.imshow(sobel_magnitude, cmap='gray') # cmap='gray' 사용
plt.title('Sobel Edge Magnitude')
plt.axis('off')

plt.show()