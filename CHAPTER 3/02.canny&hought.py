import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 에지 맵 생성
img = cv.imread('dabo.jpg')
# cv.Canny()를 사용하여 노이즈를 제거하고 얇은 선 형태의 에지 맵을 생성
# 에지인지 판단하는 최소/최대 임계값 threshold1과 threshold2는 100과 200으로 설정
edges = cv.Canny(img, 100, 200)

# 2. 허프 변환을 사용하여 이미지에서 직선 검출
# 힌트: rho : 거리 픽셀
# theta : 각도 1도 단위
# threshold : 직선으로 인정받기 위해 필요한 최소 투표수
# minLineLength : 이 길이보다 짧은 선은 무시하여 자잘한 노이즈 제거
# maxLineGap : 끊어진 선 사이의 간격이 10 이내면 하나의 선으로 이음 

lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                        minLineLength=50, maxLineGap=10)

# 3. 원본 이미지를 복사하여 그 위에 검출된 직선들을 그림
img_draw = img.copy()
# 검출된 직선 리스트가 비어있지 않은 경우에만 반복문을 실행
if lines is not None:
    for line in lines:
        # 추출된 각 직선의 시작점(x1, y1)과 끝점(x2, y2) 좌표를 가져옴
        x1, y1, x2, y2 = line[0]
        # 힌트: 색상은 (0, 0, 255)(빨간색), 두께는 2로 설정 
        cv.line(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 4. Matplotlib를 사용하여 결과 시각화
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