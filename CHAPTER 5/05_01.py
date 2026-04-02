import tensorflow as tf
from tensorflow.keras import layers, models

# 1. MNIST 데이터셋 로드 [cite: 15, 19]
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리: 0~255 사이의 픽셀 값을 0~1 사이로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Sequential 모델과 Dense 레이어를 활용하여 신경망 구성 [cite: 17, 20]
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

# 4. 모델 훈련 [cite: 18]
print("--- MNIST 모델 훈련 시작 ---")
model.fit(x_train, y_train, epochs=5)

# 5. 정확도 평가 [cite: 18]
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')