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