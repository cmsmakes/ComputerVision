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

# 이미지 생성
img = cv.imread('soccer.jpg')
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
![Uploading 스크린샷 2026-03-05 161618.png…]()


<핵심 기술 사용>
cv.circle(img, (x, y), brush_size, color, -1) : 클릭한 위치에 원을 그려 준다.
cv.setMouseCallback('Painting', paint) : 모든 마우스 이벤트를 이 콜백함수를 통해 처리를 해준다.
event == cv.EVENT_LBUTTONDOWN : 좌 클릭 시작
event == cv.EVENT_RBUTTONDOWN : 우 클릭 시작
event == cv.EVENT_MOUSEMOVE : 마우스 이동( 드래그 중 계속 그리기 )
event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP : 클릭 종료( 그리기 종료 )
key = cv.waitKey(1) & 0xFF : 루프 안에서 키보드 입력 처리

---

01_03.py ( 마우스로 영역 선택 및 ROI(관심영역) 추출 )
이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역(ROI)을 선택하여 선택한 영역만 따로 저장하거나 표시를 할 수 있게 한다.

<전체코드>
```python

import cv2 as cv

# 전역 변수 초기화
drawing = False # 드래그 상태를 확인하는 변수
start_x, start_y = -1, -1   # 드래그 시작점 좌표 초기화
img = None      # 원본 이미지 변수 초기화
temp_img = None # 드래그 중 시각화를 위한 임시 이미지 변수 초기화
roi = None      # 선택된 영역(ROI) 저장 변수 초기화

def select_roi(event, x, y, flags, param):                  # 마우스 이벤트에 따라 ROI 선택 동작 수행
    global drawing, start_x, start_y, img, temp_img, roi    # 전역 변수 사용 선언

    if event == cv.EVENT_LBUTTONDOWN:   # 좌클릭 시작: 드래그 시작
        drawing = True                  # 드래그 상태로 전환
        start_x, start_y = x, y         # 클릭한 시작점 저장

    elif event == cv.EVENT_MOUSEMOVE:   # 마우스 이동: 드래그 중이면 선택 영역 시각화
        if drawing:                     # 드래그 중일 때만 시각화
            temp_img = img.copy()       # 원본 훼손을 막기 위해 복사본에 그리기
            cv.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2) # 드래그 영역 시각화
            cv.imshow('ROI Selection', temp_img)                               # 드래그 중인 영역을 보여주는 창 업데이트

    elif event == cv.EVENT_LBUTTONUP:   # 좌클릭 종료: 드래그 종료 및 ROI 확정
        drawing = False                 # 드래그 상태 종료
        cv.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)  # 최종 선택 영역 시각화
        cv.imshow('ROI Selection', temp_img)    # 최종 선택 영역을 보여주는 창 업데이트

        # 시작점과 끝점의 좌표를 정렬 (우하단에서 좌상단으로 드래그할 경우 대비)
        x1, x2 = min(start_x, x), max(start_x, x)   # x 좌표 정렬
        y1, y2 = min(start_y, y), max(start_y, y)   # y 좌표 정렬
        
        # 유효한 영역이 선택되었는지 확인 후 ROI 추출 (Numpy 슬라이싱) 
        if x2 - x1 > 0 and y2 - y1 > 0: # 선택된 영역이 유효한 크기인지 확인
            roi = img[y1:y2, x1:x2]     # 선택된 영역(ROI) 추출
            cv.imshow('ROI', roi)       # 잘라낸 영역을 별도의 창에 출력 

# 1. 이미지 로드 
img = cv.imread('girl_laughing.jpg')    # 이미지 업로드
if img is None:                         # 이미지가 제대로 불러와졌는지 확인
    print("이미지를 찾을 수 없습니다.")
    exit()
    
temp_img = img.copy()                   # 드래그 중 시각화를 위한 임시 이미지 초기화

cv.namedWindow('ROI Selection')         # ROI 선택 창 생성
cv.setMouseCallback('ROI Selection', select_roi) # 마우스 콜백 함수 등록

print("'r': 영역 선택 리셋, 's': ROI 저장, 'q': 종료")

while True:
    cv.imshow('ROI Selection', temp_img)    # ROI 선택 창에 이미지 표시
    key = cv.waitKey(1) & 0xFF              # 루프 안에서 키보드 입력 처리

    if key == ord('r'):         # r 키: 영역 선택 리셋 
        temp_img = img.copy()   # 원본 이미지로 초기화하여 선택 영역 리셋
        if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 1:   # ROI 창이 열려 있으면 닫기
            cv.destroyWindow('ROI')                                 # 기존 ROI 창 닫기
    elif key == ord('s'):       # s 키: 선택한 영역(ROI) 저장 
        if roi is not None:     # 유효한 ROI가 선택된 경우에만 저장
            cv.imwrite('saved_roi.jpg', roi) # 이미지 파일로 저장 
            print("ROI가 'saved_roi.jpg'로 저장되었습니다.")
        else:                   # 유효한 ROI가 선택되지 않은 경우 경고 메시지 출력
            print("저장할 영역이 선택되지 않았습니다.")
    elif key == ord('q'):       # q 키: 창 종료
        break

cv.destroyAllWindows()          # 모든 창 닫기

```

<실행 결과화면>
<img width="1442" height="992" alt="스크린샷 2026-03-05 153610" src="https://github.com/user-attachments/assets/cde4acaa-a34a-4f86-a5fa-6d42ecd04e72" />

<핵심 기술 사용>
cv.setMouseCallback('ROI Selection', select_roi) : 'ROI Selection'이라는 창에서 발생하는 모든 마우스 이벤트(클릭, 이동, 떼기 등)를 사용자가 정의한 select_roi 함수로 전달합니다. 이 줄이 없으면 마우스를 아무리 움직여도 프로그램은 반응하지 않습니다.
roi = img[y1:y2, x1:x2] : 선택한 영역(ROI) 추출합니다
temp_img = img.copy() 
cv.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2) : 드래그하는 동안 사각형이 실시간으로 그려지게 만드는 테크닉입니다. (복사를 하는 이유 : 원본 img에 직접 사각형을 그려버리면, 마우스를 움직일 때마다 이전 위치에 그려진 사각형 잔상이 지워지지 않고 계속 남기 때문입니다.)
