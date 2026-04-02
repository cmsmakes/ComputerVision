import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. CIFAR-10 데이터셋 로드 [cite: 33, 41]
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리: 정규화 (0~1 범위) [cite: 34, 43]
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. CNN 모델 설계 [cite: 35, 42]
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

# 4. 모델 컴파일 및 훈련 [cite: 35]
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