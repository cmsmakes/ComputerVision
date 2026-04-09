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