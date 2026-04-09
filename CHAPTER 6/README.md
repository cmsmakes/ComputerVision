
## 🚀01.SORT 알고리즘을 활용한 다중 객체 추적기 구현
### YOLOv3 검출기와 SORT 알고리즘(칼만 필터, 헝가리안 알고리즘)을 결합하여 비디오 내 다수 객체에 고유 ID를 부여하고 실시간으로 추적하는 시스템을 구현하는 것입니다.


**전체코드**

``` python

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment # 헝가리안 알고리즘용

# 1. YOLOv3 모델 로드
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 2. Kalman Filter를 포함한 객체 추적 단위
class KalmanTracker:
    def __init__(self, bbox, track_id):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        
        x, y, w, h = bbox
        self.kf.statePost = np.array([x+w/2, y+h/2, w*h, w/float(h), 0, 0, 0], np.float32)
        self.id = track_id
        self.time_since_update = 0 # 마지막 업데이트 이후 경과된 프레임 수

    def predict(self):
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, bbox):
        self.time_since_update = 0 # 업데이트 시 경과 시간 초기화
        x, y, w, h = bbox
        measurement = np.array([x+w/2, y+h/2, w*h, w/float(h)], np.float32)
        self.kf.correct(measurement)

# 3. SORT 추적기 클래스 (수명 주기 관리 포함)
class SortTracker:
    def __init__(self):
        self.trackers = []
        self.id_count = 0
        self.max_age = 10 # 10프레임 동안 검출 안 되면 삭제

    def update(self, detections):
        # 1) 기존 추적기 위치 예측
        for t in self.trackers:
            t.predict()

        # 2) 데이터 연관 (Hungarian Algorithm)
        if len(self.trackers) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(detections), len(self.trackers)))
            for d, det in enumerate(detections):
                for t, trk in enumerate(self.trackers):
                    pred = trk.kf.statePre
                    dist = np.hypot(det[0]+det[2]/2 - pred[0], det[1]+det[3]/2 - pred[1])
                    cost_matrix[d, t] = dist

            det_indices, trk_indices = linear_sum_assignment(cost_matrix)
            
            matched_indices = []
            for d, t in zip(det_indices, trk_indices):
                if cost_matrix[d, t] < 60: # 거리 임계값
                    self.trackers[t].update(detections[d])
                    matched_indices.append(d)

            for i in range(len(detections)):
                if i not in matched_indices:
                    self.trackers.append(KalmanTracker(detections[i], self.id_count))
                    self.id_count += 1
        else:
            for det in detections:
                self.trackers.append(KalmanTracker(det, self.id_count))
                self.id_count += 1

        # 3) 수명 주기 관리: 오래된 추적기 삭제
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 4) 유효한 결과만 반환
        res = []
        for t in self.trackers:
            # 방금 업데이트된 따끈따끈한 객체만 화면에 표시 (선택 사항)
            if t.time_since_update > 0: continue 

            pos = t.kf.statePost
            if np.any(np.isnan(pos)) or np.any(np.isinf(pos)): continue
            
            area_ratio_prod = pos[2] * pos[3]
            if area_ratio_prod <= 0: continue

            w = np.sqrt(area_ratio_prod)
            h = pos[2] / w
            
            try:
                res.append([int(pos[0]-w/2), int(pos[1]-h/2), int(w), int(h), t.id])
            except (ValueError, OverflowError):
                continue
        return res

tracker = SortTracker()
cap = cv2.VideoCapture("slow_traffic_small.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > 0.5:
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(detection[0] * width - w/2), int(detection[1] * height - h/2)
                detections.append([x, y, w, h])

    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x, y, w, h, track_id = obj
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("SORT Tracking", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()

```

**실행 결과**

<img width="642" height="392" alt="스크린샷 2026-04-09 124326" src="https://github.com/user-attachments/assets/6fd0d0a1-1b66-4b53-bb2d-b6e7dfefd20c" />



**💡 핵심 기술 요약**

**`self.kf.transitionMatrix 및 self.kf.predict()`**: 객체의 상태를 [중심x, 중심y, 면적, 가로세로비, 속도vx, 속도vy, 속도vs]로 정의하여, 다음 프레임에서 객체가 어디에 있을지 미리 예측합니다. 이를 통해 일시적인 검출 누락에도 유연하게 대응할 수 있습니다.

**`linear_sum_assignment(cost_matrix)`**: 모든 검출값과 추적기 사이의 거리를 계산한 비용 행렬(Cost Matrix)을 바탕으로, 전체 오차의 합이 최소가 되도록 1:1 매칭을 수행합니다. 이는 객체가 서로 교차하거나 근접할 때 ID가 바뀌는 현상을 방지합니다.

**`self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]`**: max_age 변수를 도입하여 일정 프레임(예: 10프레임) 동안 검출되지 않은 추적기는 화면 밖으로 나간 것으로 간주하고 리스트에서 삭제합니다. 이전 코드에서 박스가 무한히 생기던 문제를 해결하는 결정적인 부분입니다.

**`np.any(np.isnan(pos)), if t.time_since_update > 0: continue, cv2.putText`**: 현재 프레임에서 실제로 매칭에 성공한(time_since_update == 0) 객체만 화면에 그려 시각적 노이즈를 최소화합니다.

---

## 🚀02. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화
### Mediapipe FaceMesh를 활용해 얼굴의 468개 랜드마크를 정밀하게 검출하고, 이를 실시간 웹캠 영상 좌표에 맞춰 시각화하는 프로그램을 만드는 것입니다.


**전체코드**
```python

import cv2
import sys
import os

# 1. Mediapipe FaceLandmarker 초기화 (새로운 Tasks API 사용)
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
except ImportError:
    print("Mediapipe 라이브러리를 찾을 수 없습니다. 아래 명령어로 설치해 주세요:")
    print("  pip install mediapipe")
    sys.exit()

# 모델 파일 경로 설정
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
if not os.path.exists(model_path):
    print(f"모델 파일을 찾을 수 없습니다: {model_path}")
    print("아래 명령어로 모델을 다운로드해 주세요:")
    print('  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"')
    sys.exit()

# 모델 파일 읽기 (한글 경로 문제 방지를 위해 파일 데이터를 직접 메모리에 로드)
with open(model_path, 'rb') as f:
    model_data = f.read()

# FaceLandmarker 옵션 설정
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=model_data),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

face_landmarker = FaceLandmarker.create_from_options(options)

# 2. 웹캠 연결
cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 처리 속도 향상을 위해 RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mediapipe Image 객체 생성 및 랜드마크 검출
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 33  # 약 30fps 기준 타임스탬프 증가

    # 랜드마크 시각화
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            ih, iw, _ = image.shape
            for lm in face_landmarks:
                # 정규화된 좌표를 이미지 크기에 맞게 변환
                x, y = int(lm.x * iw), int(lm.y * ih)
                # 각 랜드마크 위치에 점 표시
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Mediapipe FaceMesh - Homework 02', image)

    # 3. ESC 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
face_landmarker.close()
cv2.destroyAllWindows()

```

**실행 결과화면**

<img width="642" height="512" alt="스크린샷 2026-04-09 132528" src="https://github.com/user-attachments/assets/f46f8e14-281d-4619-81dc-cb66a1acc633" />



**💡 핵심 기술 요약**

**`FaceLandmarkerOptions`**: 검출할 얼굴의 수(num_faces=1)와 검출 신뢰도 등을 설정하며, 특히 실시간 분석을 위해 RunningMode.VIDEO 모드를 지정합니다.

**`model_asset_buffer=model_data`**: 468개 이상의 얼굴 랜드마크 데이터가 포함된 face_landmarker.task 모델 파일을 메모리에 로드하여 검출기를 생성합니다

**`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`**: OpenCV는 기본적으로 BGR 형식을 사용하지만, Mediapipe는 RGB 형식을 요구하므로 색상 채널을 변환합니다.

**`face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)`** : 비디오 스트림의 타임스탬프와 함께 이미지를 입력하여 실시간으로 얼굴 랜드마크 좌표를 계산합니다

**`x, y = int(lm.x * iw), int(lm.y * ih)`** : 이미지의 가로 폭(iw)과 세로 높이(ih)를 정규화된 좌표(lm.x, lm.y)에 곱하여 실제 점을 찍을 위치를 구합니다.

**`cv2.circle(image, (x, y), 1, (0, 255, 0), -1)`** : 계산된 좌표에 초록색 점을 그려 얼굴의 형태(Mesh)가 실시간으로 보이게 합니다.

**`cv2.waitKey(5) & 0xFF == 27`** : 사용자가 ESC 키를 누르는 순간 프로그램을 안전하게 종료하도록 설정합니다.
