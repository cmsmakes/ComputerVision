import cv2 as cv
import numpy as np

# 1. 이미지 로드 (soccer.jpg 자리에 본인이 가진 이미지 파일 이름을 넣으세요)

img = cv.imread('soccer.jpg') # soccer.jpg 이미지 업로드

img = cv.resize(img, (0, 0), fx=0.5, fy=0.5) # 이미지 크기 조절 (절반으로 줄임)

# 이미지가 제대로 불러와졌는지 확인
if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 2. 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지 그레이스케일로 변환

# 3. 가로로 연결(np.hstack)하기 위해 그레이스케일 이미지를 3채널로 임시 변환
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 4. 원본 이미지와 그레이스케일 이미지를 가로로 연결
result = np.hstack((img, gray_3ch)) 

# 5. 결과를 화면에 표시하고, 아무 키나 누르면 창 닫기
cv.imshow('Original vs Grayscale', result) # 원본과 그레이스케일 이미지를 나란히 보여주는 창 생성
cv.waitKey(0) # 키 입력 대기
cv.destroyAllWindows() # 모든 창 닫기