import cv2 as cv
from matplotlib import pyplot as plt

# 1. 이미지 불러오기 [cite: 17, 34]
img = cv.imread('mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 흑백 이미지 생성

# 2. SIFT 객체 생성 [cite: 20]
sift = cv.SIFT_create(nfeatures=500)

# 3. 특징점(Keypoints) 검출 [cite: 21]
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 4. 특징점 시각화 (배경을 흑백인 gray로 설정) [cite: 22, 26]
# 첫 번째 인자를 gray로 바꾸면 특징점은 컬러로, 배경은 흑백으로 나옵니다.
img_with_keypoints = cv.drawKeypoints(gray, keypoints, None, 
                                      flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 5. 결과 출력 [cite: 23]
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