## 🚀01.SIFT를 이용한 특징점 검출 및 시각화
### 이미지 내 SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 특징점을 검출하고 시각화하기

### 핵심 윈리 : SIFT는 단순히 점을 찾는 것을 넘어, 이미지가 회전하거나 크기(Scale)가 변해도 동일한 지점을 찾아낼 수 있는 강력한 지역 특징(Local Feature) 추출 알고리즘입니다.
1. 스케일 불변성 (Scale Invariance): 이미지를 다양한 크기(옥타브)로 만들고 점점 흐리게(Gaussian Blur) 하여, 멀리서 보든 가까이서 보든 상관없이 공통된 특징을 찾습니다.
2. 회전 불변성 (Rotation Invariance): 특징점 주변의 밝기 변화(Gradient) 방향을 분석하여 가장 지배적인 방향을 '기준 방향'으로 설정합니다.
3. 특징 기술자 (Descriptor): 각 특징점 주변의 정보를 128차원 벡터로 수치화하여, 다른 이미지의 특징점과 비교(매칭)할 수 있는 '지문'과 같은 데이터를 생성합니다.

**전체코드**

``` python

import cv2 as cv
from matplotlib import pyplot as plt

# 1. 이미지 불러오기 
img = cv.imread('mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 흑백 이미지 생성

# 2. SIFT 객체 생성 
sift = cv.SIFT_create(nfeatures=5000)

# 3. 특징점(Keypoints) 검출 
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 4. 특징점 시각화 (배경을 흑백인 gray로 설정) 
# 첫 번째 인자를 gray로 바꾸면 특징점은 컬러로, 배경은 흑백으로 나옵니다.
img_with_keypoints = cv.drawKeypoints(gray, keypoints, None, 
                                      flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 5. 결과 출력 
plt.figure(figsize=(12, 6))

# 왼쪽: 원본 컬러 이미지
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 오른쪽: 흑백 배경 위의 특징점
plt.subplot(1, 2, 2)
# drawKeypoints의 결과는 항상 BGR 형태이므로 다시 RGB로 바꿔줍니다.
plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
plt.title('SIFT Keypoints on Gray')
plt.axis('off')

plt.show()

```

**실행 결과**

<img width="945" height="282" alt="image" src="https://github.com/user-attachments/assets/70d9b773-2eff-4d1a-aa48-107b38a1c543" />



**💡 핵심 기술 요약**

**`sift = cv.SIFT_create(nfeatures=5000)`**: SIFT 연산을 수행할 엔진을 만듭니다.

**`keypoints, descriptors = sift.detectAndCompute(gray, None)`**: 가장 핵심적인 함수로, 이미지에서 특징점의 위치(keypoints)를 찾고, 그 지점의 고유한 수치 정보(descriptors)를 동시에 계산합니다.

**`img_with_keypoints = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)`**: DRAW_RICH_KEYPOINTS를 사용하면 단순한 점이 아니라, **원의 크기(Scale)**와 **원 안의 선(Orientation)**을 통해 SIFT가 파악한 기하학적 정보를 시각적으로 보여줍니다.

---

## 🚀02.SIFT를 이용한 두 영상 간 특징점 매칭
### 두 이미지를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화

### 핵심 윈리 : 같은 물체의 같은 곳을 쌍으로 맺는 일
1. 변환 불변성: 물체가 이동, 회전하거나 크기(Scale)가 변해도 동일한 지점을 찾아내어 연결할 수 있어야 합니다.
2. 분별력: 물체의 다른 곳에서 추출된 특징과는 두드러지게 달라야 엉뚱한 곳과 매칭되지 않습니다.
3. 거리 계산: 특징 기술자(Descriptor) 사이의 거리를 계산하여 값이 작을수록 서로 같은 특징점으로 판단합니다.


**전체코드**
```python

import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color80.jpg')

# 이미지를 정상적으로 읽었는지 확인
if img1 is None or img2 is None:
    print("이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    # 2. SIFT 객체 생성 및 특징점 추출 
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. BFMatcher 객체 생성 (L2 거리 사용 및 crossCheck 활성화) 
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    # 4. 특징점 매칭 수행 
    matches = bf.match(des1, des2)

    # 매칭 결과를 거리 순으로 정렬 (거리가 짧을수록 유사도가 높음)
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. 매칭 결과 시각화 
    # 상위 500개의 매칭점만 표시
    res = cv.drawMatches(img1, kp1, img2, kp2, matches[:500], None, 
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 6. Matplotlib을 이용한 결과 출력 
    plt.figure(figsize=(12, 6))
    plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
    plt.title('SIFT Feature Matching (Top 500 Matches)')
    plt.axis('off')
    plt.show()

```

**실행 결과화면**

<img width="973" height="321" alt="image" src="https://github.com/user-attachments/assets/bc82e31d-dfd8-4d7f-b215-7ab28cefe686" />



**💡 핵심 기술 요약**

**`kp1, des1 = sift.detectAndCompute(img1, None)`**: 각 특징점 주변의 16x16 영역을 4x4 블록으로 나누어 128차원 벡터를 생성합니다. 그리고 영상의 한 지점을 고유하게 식별할 수 있는 **'지문'**과 같은 데이터를 만드는 단계입니다.

**`bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)`**: crossCheck=True: 서로가 서로에게 최선의 매칭일 때만 남기는 기능으로, 매칭의 정확도를 높여줍니다.

**`matches = sorted(matches, key=lambda x: x.distance)`**: 거리가 짧을수록 두 지점의 유사도가 높다는 것을 의미하며, 상위 결과만 선택하여 **이상치(Outlier)**를 줄일 수 있습니다.

**`res = cv.drawMatches(img1, kp1, img2, kp2, matches[:500], ...)`** : 두 영상을 가로로 붙이고 대응되는 특징점들을 선으로 연결하여 보여줍니다.

---

## 🚀03.호모그래피를 이용한 이미지 정합 (Image Alignment)
###  SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬

### 핵심 원리 : 서로 다른 각도에서 찍은 두 평면 사이의 투시 변환(Perspective Transformation) 관계를 계산하는 것
1. 호모그래피(Homography): 한 평면을 다른 평면으로 투영시킬 때 사용하는 $3 \times 3$ 행렬입니다. 최소 4쌍의 대응점이 있으면 계산이 가능합니다.
2. Ratio Test: 1등 매칭 거리와 2등 매칭 거리의 비율을 계산하여(예: 0.7 미만), 모호한 매칭을 사전에 제거함으로써 정확도를 높입니다.
3. RANSAC (Random Sample Consensus): 무작위로 샘플을 뽑아 모델을 만든 후 검증하는 과정을 반복하여, 잘못 매칭된 점(Outlier)들을 배제하고 가장 신뢰도 높은 변환 행렬을 찾아냅니다.

**전체코드**
```python

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1. 이미지 불러오기 
img1 = cv.imread('img1.jpg') 
img2 = cv.imread('img2.jpg') 

# 2. SIFT 특징점 검출 
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3. FLANN 매처 및 Ratio Test 적용 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

if len(good_matches) > 10:
    # --- 데이터 준비 1: 특징점 매칭 시각화 --- 
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, 
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

    # --- 데이터 준비 2: 이미지 정합 (Panorama) ---
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 호모그래피 계산 
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 이미지 변환 및 합성 
    warped_img = cv.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
    result = warped_img.copy()
    result[0:h1, 0:w1] = img1
    result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    # --- 최종 출력: 하나의 Figure에 두 Plot 배치 --- 
    plt.figure(figsize=(12, 10)) # 전체 창 크기 설정

    # 첫 번째 칸 (위): 매칭 결과
    plt.subplot(2, 1, 1) # 2행 1열 중 1번째
    plt.imshow(img_matches_rgb)
    plt.title('Step 1: Feature Matching (Correspondences)')
    plt.axis('off')

    # 두 번째 칸 (아래): 정합 결과
    plt.subplot(2, 1, 2) # 2행 1열 중 2번째
    plt.imshow(result_rgb)
    plt.title('Step 2: Final Stitched Panorama')
    plt.axis('off')

    plt.tight_layout() # 이미지 간 간격 자동 조정
    plt.show()

else:
    print("충분한 매칭점을 찾지 못했습니다.")

```

**실행 결과화면**


<img width="1168" height="918" alt="image" src="https://github.com/user-attachments/assets/2cf04a69-96c9-4b0c-85c6-5159f2ec16df" />




**💡 핵심 기술 요약**

**`flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]`**: 수만 개의 특징점을 빠르게 매칭하기 위해 KD-Tree 구조를 사용하는 FLANN 라이브러리를 활용합니다. 그리고 0.7이라는 임계값을 사용한 Ratio Test를 통해 매칭의 품질을 보장합니다

**`H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)`**: cv.RANSAC 플래그를 통해 잘못된 대응점(Outlier)이 섞여 있어도 강인하게(Robust) 올바른 행렬을 찾아냅니다.

**`warped_img = cv.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
result[0:h1, 0:w1] = img1`**: 계산된 호모그래피 행렬 H를 이용하여 img2를 img1의 좌표계로 찌그러뜨려 변환합니다.

