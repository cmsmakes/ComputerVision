
## 🚀01.SORT 알고리즘을 활용한 다중 객체 추적기 구현
### 


**전체코드**

``` python

import cv2
import numpy as np

# 1. YOLOv3 모델 로드
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 2. 아주 간단한 추적기 클래스 (SORT 개념의 최소 구현)
class SimpleTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, detections):
        objects_bbs_ids = []
        for rect in detections:
            x, y, w, h, conf = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = np.hypot(cx - pt[0], cy - pt[1])
                if dist < 35: # 이전 프레임과 중심점 거리가 가까우면 같은 객체로 판단
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, x+w, y+h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, x+w, y+h, self.id_count])
                self.id_count += 1

        # 사용되지 않는 ID 정리
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

tracker = SimpleTracker()

# 비디오 로드
cap = cv2.VideoCapture("slow_traffic_small.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detections.append([x, y, w, h, confidence])

    # 3. 추적 업데이트
    track_bbs_ids = tracker.update(detections)

    # 4. 시각화
    for track in track_bbs_ids:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Multi-Object Tracking", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()

```

**실행 결과**

<img width="642" height="392" alt="스크린샷 2026-04-09 124326" src="https://github.com/user-attachments/assets/6fd0d0a1-1b66-4b53-bb2d-b6e7dfefd20c" />



**💡 핵심 기술 요약**

**``**: 

**``**: 

**``**: 

---

## 🚀02. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화
### 


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

**``**: 

**``**: 

**``**: 

**``** : 

**``** : 

**``** : 
