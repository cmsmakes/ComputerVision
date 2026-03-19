import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 에지 맵 생성 [cite: 44]
img = cv.imread('dabo.jpg')
# 힌트: cv.Canny()에서 threshold1과 threshold2는 100과 200으로 설정 [cite: 49]
edges = cv.Canny(img, 100, 200)

# 2. 허프 변환을 사용하여 이미지에서 직선 검출 [cite: 45]
# 힌트: rho, theta, threshold, minLineLength, maxLineGap 값을 조정 
# 선이 너무 파편화된다면 minLineLength를 낮추고, maxLineGap을 높여보세요.
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                        minLineLength=50, maxLineGap=10)

# 3. 검출된 직선을 원본 이미지에 그리기 [cite: 46]
img_draw = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 힌트: 색상은 (0, 0, 255)(빨간색), 두께는 2로 설정 [cite: 51]
        cv.line(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 4. Matplotlib를 사용하여 결과 시각화 [cite: 47]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_draw, cv.COLOR_BGR2RGB))
plt.title('Edge Lines (Hough)')
plt.axis('off')

plt.show()