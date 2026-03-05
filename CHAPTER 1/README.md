# CHAPTER 1 과제 설명

01_01.py (이미지 불러오기 및 그레이스케일 변환)
OpenCV를 사용하여 이미지를 불러오고 화면에 출력 -> 원본 이미지와 그레이스케일로 변환된 이미지를 하나의 창으로 나란히 표시

<전체코드>
``` python
import cv2 as cv
import numpy as np

# 1. 이미지 로드 (soccer.jpg 자리에 본인이 가진 이미지 파일 이름을 넣으세요)

img = cv.imread('soccer.jpg') # soccer.jpg 이미지 업로드

img = cv.resize(img, (0, 0), fx=0.5, fy=0.5) # 이미지 크기 조절 (절반으로 줄임)

# 이미지가 제대로 불러와졌는지 확인
if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 2. 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지 그레이스케일로 변환

# 3. 가로로 연결(np.hstack)하기 위해 그레이스케일 이미지를 3채널로 임시 변환
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 4. 원본 이미지와 그레이스케일 이미지를 가로로 연결
result = np.hstack((img, gray_3ch)) 

# 5. 결과를 화면에 표시하고, 아무 키나 누르면 창 닫기
cv.imshow('Original vs Grayscale', result) # 원본과 그레이스케일 이미지를 나란히 보여주는 창 생성
cv.waitKey(0) # 키 입력 대기
cv.destroyAllWindows() # 모든 창 닫기

```
<실행 결과화면>
<img width="1436" height="506" alt="스크린샷 2026-03-05 153442" src="https://github.com/user-attachments/assets/454839df-f02b-46d5-8590-822429a299a3" />

<핵심 기술 사용>
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) : 이미지를 그레이스케일로 변환하는 코드이다.
result = np.hstack((img, gray_3ch))  : 원본 이미지와 그레이스케일 이미지를 가로로 연결
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) : 가로로 연결(np.hstack)하기 위해 그레이스케일 이미지를 3채널로 임시 변환해야한다 ( 이 코드가 없을 시 오류 발생을 한다 )

---

01_02.py ( 페인팅 붓 크기 조절 기능 추가 )
마우스 입력으로 이미지 위에 붓질이 가능하며 좌클릭시 파란색이 나오며, 우클릭시 빨간색이 나오도록 하였으며 드래그로 연속 그리기도 가능하다. 붓 크기 조절은 "+" 키를 누르게 되면 붓 크기가 커지고, "-" 키를 누르게 되면 붓 크기가 작아지며 붓 크기는 최소 1, 최대 15로 제한을 하였다. 실행창 종료는 "q"로 종료를 하도록 만들었다.

<전체코드>
```python

import cv2 as cv
import numpy as np

# 초기 설정
brush_size = 5 # 초기 붓 크기 
drawing = False # 드래그 상태를 확인하는 변수 
color = (255, 0, 0) # 기본 색상 (파란색, OpenCV는 BGR 형식 사용)

# 마우스 이벤트 처리 함수
def paint(event, x, y, flags, param): # 마우스 이벤트에 따라 그리기 동작 수행
    global drawing, color, brush_size # 전역 변수 사용 선언

    if event == cv.EVENT_LBUTTONDOWN: # 좌클릭 시작: 그리기 시작
        drawing = True                # 그리기 상태로 전환
        color = (255, 0, 0)           # 좌클릭: 파란색
        cv.circle(img, (x, y), brush_size, color, -1) # 클릭한 위치에 원 그리기
    elif event == cv.EVENT_RBUTTONDOWN:    # 우클릭 시작: 그리기 시작
        drawing = True                     # 그리기 상태로 전환
        color = (0, 0, 255)                # 우클릭: 빨간색
        cv.circle(img, (x, y), brush_size, color, -1) # 클릭한 위치에 원 그리기
    elif event == cv.EVENT_MOUSEMOVE:        # 마우스 이동: 드래그 중이면 계속 그리기
        if drawing: # 드래그로 연속 그리기 
            cv.circle(img, (x, y), brush_size, color, -1)
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP: # 클릭 종료: 그리기 중지
        drawing = False # 마우스를 떼면 그리기 중지

# 1000x1000 크기의 흰색 캔버스(이미지) 생성
img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
cv.namedWindow('Painting') # 그리기 창 생성
cv.setMouseCallback('Painting', paint) # 마우스 콜백 함수 등록

print("'+'/'=': 붓 크기 증가, '-': 붓 크기 감소, 'q': 종료")

while True:
    cv.imshow('Painting', img)  # 그리기 창에 이미지 표시
    key = cv.waitKey(1) & 0xFF # 루프 안에서 키보드 입력 처리

    if key == ord('q'): # q 키: 창 종료
        break
    elif key == ord('+') or key == ord('='):  # + 키 (또는 = 키): 붓 크기 증가
        brush_size = min(15, brush_size + 1)  # 최대 15로 제한
        print(f"현재 붓 크기: {brush_size}")    # 현재 붓 크기 출력
    elif key == ord('-'):                     # - 키: 붓 크기 감소
        brush_size = max(1, brush_size - 1)   # 최소 1로 제한
        print(f"현재 붓 크기: {brush_size}")    # 현재 붓 크기 출력

cv.destroyAllWindows()   # 모든 창 닫기

```

<실행 결과화면>
<img width="1002" height="1032" alt="스크린샷 2026-03-05 153537" src="https://github.com/user-attachments/assets/10b7c0f6-7335-4ba0-ad6f-15d1be9840ac" />

<핵심 기술 사용>
cv.circle(img, (x, y), brush_size, color, -1) : 클릭한 위치에 원을 그려 준다.
cv.setMouseCallback('Painting', paint) : 모든 마우스 이벤트를 이 콜백함수를 통해 처리를 해준다.
event == cv.EVENT_LBUTTONDOWN : 좌 클릭 시작
event == cv.EVENT_RBUTTONDOWN : 우 클릭 시작
event == cv.EVENT_MOUSEMOVE : 마우스 이동( 드래그 중 계속 그리기 )
event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP : 클릭 종료( 그리기 종료 )
key = cv.waitKey(1) & 0xFF : 루프 안에서 키보드 입력 처리
