import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. cv.imread()를 사용하여 이미지 불러오기 [cite: 19]
img = cv.imread('edgeDetectionImage.jpg') 

# 2. cv.cvtColor()를 사용하여 그레이스케일로 변환 [cite: 20]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 3. cv.Sobel()을 사용하여 x축과 y축 방향 에지 검출 [cite: 21, 27]
# ksize는 3 또는 5로 설정 [cite: 27]
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 4. cv.magnitude()를 사용하여 에지 강도 계산 [cite: 22]
magnitude = cv.magnitude(sobel_x, sobel_y)

# 5. cv.convertScaleAbs()를 사용하여 uint8로 변환 [cite: 28]
sobel_magnitude = cv.convertScaleAbs(magnitude)

# 6. Matplotlib를 사용하여 시각화 [cite: 23, 29]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel_magnitude, cmap='gray') # cmap='gray' 사용 [cite: 29]
plt.title('Sobel Edge Magnitude')
plt.axis('off')

plt.show()