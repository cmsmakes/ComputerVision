## 🚀01.간단한 이미지 분류기 구현
### 손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현이미지 분류기 구현


**전체코드**

``` python

import tensorflow as tf
from tensorflow.keras import layers, models

# 1. MNIST 데이터셋 로드 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리: 0~255 사이의 픽셀 값을 0~1 사이로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Sequential 모델과 Dense 레이어를 활용하여 신경망 구성 
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # 2D 이미지를 1D 벡터로 평탄화
    layers.Dense(128, activation='relu'), # 은닉층
    layers.Dropout(0.2),                  # 과적합 방지
    layers.Dense(10, activation='softmax') # 출력층 (0~9까지 10개 클래스)
])

# 3. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 모델 훈련 
print("--- MNIST 모델 훈련 시작 ---")
model.fit(x_train, y_train, epochs=5)

# 5. 정확도 평가 
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')

```

**실행 결과**

<img width="465" height="233" alt="스크린샷 2026-04-02 155217" src="https://github.com/user-attachments/assets/1a14f84c-9ec9-4c9a-b22c-69152eb67452" />





**💡 핵심 기술 요약**

**`layers.Flatten(input_shape=(28, 28))`**: 2차원 행렬 형태(28 X 28 픽셀)인 MNIST 데이터를 784개의 요소를 가진 1차원 벡터로 펼쳐줍니다. 그리고 다음에 오는 Dense 레이어(완전 연결 계층)는 데이터를 1열로 세워진 입력으로만 받을 수 있기 때문입니다.

**`layers.Dense(128, activation='relu') / layers.Dense(10, activation='softmax')`**: 은닉층 (Hidden Layer): 128개의 뉴런이 이미지의 복잡한 패턴을 학습합니다. relu 활성화 함수는 학습 속도를 높이고 성능을 개선하는 데 표준적으로 사용됩니다. / 은닉층 (Hidden Layer): 128개의 뉴런이 이미지의 복잡한 패턴을 학습합니다. relu 활성화 함수는 학습 속도를 높이고 성능을 개선하는 데 표준적으로 사용됩니다.

**`model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', ...)`**: 손실 함수 (Loss Function): 모델이 예측한 값과 실제 정답 사이의 오차를 계산합니다. MNIST처럼 정답이 정수 형태(0, 1, 2...)인 다중 분류 문제에서는 sparse_categorical_crossentropy가 핵심적인 역할을 합니다. 옵티마이저 (Optimizer): adam은 오차를 줄이기 위해 모델의 가중치를 효율적으로 업데이트하는 알고리즘입니다.

---

## 🚀02.CIFAR-10 데이터셋을 활용한 CNN 모델 구축
### CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수


**전체코드**
```python

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. CIFAR-10 데이터셋 로드 
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리: 정규화 (0~1 범위) 
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. CNN 모델 설계 
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. 모델 컴파일 및 훈련 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("--- CIFAR-10 모델 훈련 시작 ---")
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 5. 테스트 이미지(dog.jpg) 예측 수행 
# 주의: 'dog.jpg' 파일이 코드와 같은 경로에 있어야 합니다.
def predict_dog_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # 배치 차원 추가

        predictions = model.predict(img_array)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        predicted_class = class_names[np.argmax(predictions)]
        print(f"\n파일 '{img_path}'에 대한 예측 결과: {predicted_class}")
    except Exception as e:
        print(f"\n이미지 예측 중 오류 발생: {e}")

# 예측 실행
predict_dog_image('dog.jpg')

```

**실행 결과화면**

<img width="731" height="383" alt="스크린샷 2026-04-02 160113" src="https://github.com/user-attachments/assets/4a065432-8384-4d32-9e6c-8e08ac5082fc" />




**💡 핵심 기술 요약**

**`layers.Conv2D(32, (3, 3), activation='relu', ...)`**: 3 X 3 크기의 필터를 사용하여 이미지의 공간적 특징을 추출합니다.

**`layers.MaxPooling2D((2, 2))`**: 특징 맵의 크기를 줄여 중요한 정보만 남기고 계산량을 줄입니다. 이 과정 덕분에 이미지 내 사물의 위치가 조금 바뀌어도 정확히 인식할 수 있습니다.

**`x_train / 255.0`**: 0에서255 사이의 RGB 픽셀 값을 0에서1 사이로 정규화합니다. 이는 모델의 학습 속도(수렴)를 빠르게 만드는 핵심 기술입니다.

**`input_shape=(32, 32, 3)`** : CIFAR-10 이미지가 가로 32, 세로 32, 그리고 3개의 채널(Red, Green, Blue)로 구성되어 있음을 모델에 알려주는 필수 설정입니다.

**`image.load_img(..., target_size=(32, 32))`** : 외부 이미지를 모델이 학습한 크기인 32 X 32 로 강제 조정합니다.

**`np.expand_dims(img_array, axis=0)`** : 모델은 항상 '배치(묶음)' 단위로 데이터를 받기를 기대합니다. 이미지 한 장(32, 32, 3)을 넣더라도 (1, 32, 32, 3) 형태로 차원을 늘려주어야 오류 없이 예측이 실행됩니다.

