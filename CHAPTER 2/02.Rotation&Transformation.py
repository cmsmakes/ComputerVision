import cv2
import numpy as np

# 1. 이미지 불러오기 (테스트할 이미지 경로를 입력해주세요)
img = cv2.imread('rose.png') # 강의 자료의 장미 이미지와 같은 테스트용 이미지

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # 이미지의 높이(h)와 너비(w) 가져오기
    h, w = img.shape[:2]

    # 2. 이미지의 중심 좌표 계산
    center = (w / 2, h / 2)

    # 3. 회전 및 크기 조절 행렬 생성
    # 중심 좌표 기준으로 +30도 회전, 크기는 0.8배
    # cv2.getRotationMatrix2D() 사용
    M = cv2.getRotationMatrix2D(center, angle=30, scale=0.8)

    # 4. 평행이동 적용
    # 회전 행렬의 마지막 열(index 2) 값을 조정하여 평행이동 반영
    # x축 방향으로 +80px
    M[0, 2] += 80
    # y축 방향으로 -40px
    M[1, 2] -= 40

    # 5. 변환 적용 (cv2.warpAffine 사용)
    # 이미지에 생성한 변환 행렬 M을 적용
    result = cv2.warpAffine(img, M, (w, h))

    # 6. 결과 시각화
    # 원본 이미지와 변환된 이미지를 나란히 출력
    cv2.imshow('Original', img)
    cv2.imshow('Rotated + Scaled + Translated', result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()