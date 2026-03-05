import cv2 as cv
import numpy as np

# 1. 이미지 로드 (soccer.jpg 자리에 본인이 가진 이미지 파일 이름을 넣으세요)
img = cv.imread('soccer.jpg') # [cite: 21]

img = cv.resize(img, (0, 0), fx=0.5, fy=0.5) # [cite: 22] 사진 크기 조절

# 이미지가 제대로 불러와졌는지 확인
if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 2. 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # [cite: 22, 26]

# 3. 가로로 연결(np.hstack)하기 위해 그레이스케일 이미지를 3채널로 임시 변환
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 4. 원본 이미지와 그레이스케일 이미지를 가로로 연결
result = np.hstack((img, gray_3ch)) # [cite: 23]

# 5. 결과를 화면에 표시하고, 아무 키나 누르면 창 닫기
cv.imshow('Original vs Grayscale', result) # [cite: 24]
cv.waitKey(0) # [cite: 24]
cv.destroyAllWindows()