## 🚀01.소벨 에지 검출 및 결과 시각화 (Sobel Edge Detection)
### 이미지 내 픽셀 값의 변화량(기울기)을 통해 "경계선"의 기초 정보를 추출하는 것

### 핵심 윈리 : 이미지의 가로와 세로 방향으로 미분을 수행하여, 밝기가 급격하게 변하는 지점을 찾아낸다.

**전체코드**

``` python

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

```

**실행 결과**

<img width="802" height="248" alt="image" src="https://github.com/user-attachments/assets/b43d44d9-25fc-451f-b868-9a52ac41a5e9" />

**💡 핵심 기술 요약**

**`cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)`**: x축 방향의 미분을 수행하여 수직 방향의 에지를 검출합니다.

**`cv.magnitude(sobel_x, sobel_y)`**: x, y 양방향의 에지 강도를 합산하여 전체 에지 세기를 계산합니다.

**`cv.convertScaleAbs(magnitude)`**: 계산된 미분 값을 가시화하기 위해 8비트(uint8) 이미지 형식으로 변환합니다.

---

## 🚀02.캐니 에지 및 허프 변환 ( Canny Edge & Hough Transfrom )
### 파편화된 에지 픽셀들을 연결하여 수학적으로 의미 있는 "직선" 성분을 찾아내는 것

### 핵심 윈리 : Canny : 1. 소벨보다 정교하게 노이즈를 제거하고 가느다란 에지만 남깁니다.  2. Hough Transform : 같은 선상에 놓인 점들을 분석하여 직선의 시작점과 끝점 좌표를 계


**전체코드**
```python

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

```

**실행 결과화면**

<img width="825" height="319" alt="image" src="https://github.com/user-attachments/assets/22f62834-c687-4030-b0c3-8ef20c37ea64" />


**💡 핵심 기술 요약**

**`cv.Canny(img, 100, 200)`**: 노이즈를 제거하고 가장 선명한 에지만 남겨 이진 에지 맵을 생성합니다.

**`cv.HoughLinesP(edges, 1, np.pi/180, threshold, ...)`**: 확률적 허프 변환을 통해 에지 점들을 연결하여 직선의 시작점과 끝점 좌표를 찾아냅니다.

**`cv.line(img, pt1, pt2, (0, 0, 255), 2)`**: 검출된 좌표 데이터를 바탕으로 원본 이미지 위에 빨간색 선을 그려 시각화합니다.

---

## 🚀03.GrabCut을 이용한 영역 분할 ( Interactive Segmentation )
### 사용자의 힌트(사각형 영역)를 바탕으로 배경과 전경(객체)을 완젼히 분리하는 것

### 핵심 원리 : 그랩 컷(GrabCut) 알고리즘을 사용하여 이미지 내의 픽셀들을 "배경일 확률" 과 "객체일 확률"로 나누어 최적의 경계를 찾음

**전체코드**
```python

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

```

**실행 결과화면**


<img width="1248" height="350" alt="image" src="https://github.com/user-attachments/assets/36e24c41-555f-46fe-93cb-9869191df306" />



**💡 핵심 기술 요약**

**`cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)`**: 사각형(rect) 정보를 초기값으로 사용하여 전경과 배경을 반복적으로 분리합니다.

**`np.where((mask == 0) | (mask == 2), 0, 1)`**: GrabCut 결과 마스크에서 배경 관련 값(0, 2)은 제거하고, 객체 관련 값(1, 3)만 남겨 이진 마스크를 만듭니다.

**`img * mask[:, :, np.newaxis]`**: 원본 이미지와 이진 마스크를 곱하여 배경을 검은색으로 지우고 객체만 추출합니다.

