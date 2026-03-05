import cv2 as cv
import numpy as np

# 초기 설정
brush_size = 5 # 초기 붓 크기 [cite: 35]
drawing = False # 드래그 상태를 확인하는 변수
color = (255, 0, 0) # 기본 색상 (파란색, OpenCV는 BGR 형식 사용)

# 마우스 이벤트 처리 함수 [cite: 40]
def paint(event, x, y, flags, param):
    global drawing, color, brush_size

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        color = (255, 0, 0) # 좌클릭: 파란색 [cite: 38]
        cv.circle(img, (x, y), brush_size, color, -1) # [cite: 40]
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        color = (0, 0, 255) # 우클릭: 빨간색 [cite: 38]
        cv.circle(img, (x, y), brush_size, color, -1)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing: # 드래그로 연속 그리기 [cite: 38]
            cv.circle(img, (x, y), brush_size, color, -1)
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        drawing = False # 마우스를 떼면 그리기 중지

# 500x500 크기의 흰색 캔버스(이미지) 생성
img = np.ones((500, 500, 3), dtype=np.uint8) * 255
cv.namedWindow('Painting')
cv.setMouseCallback('Painting', paint) # 마우스 콜백 함수 등록 [cite: 40]

print("'+'/'=': 붓 크기 증가, '-': 붓 크기 감소, 'q': 종료")

while True:
    cv.imshow('Painting', img)
    key = cv.waitKey(1) & 0xFF # 루프 안에서 키보드 입력 처리 [cite: 41, 42]

    if key == ord('q'): # q 키: 창 종료 [cite: 39]
        break
    elif key == ord('+') or key == ord('='): # + 키 (또는 = 키): 붓 크기 증가 [cite: 36]
        brush_size = min(15, brush_size + 1) # 최대 15로 제한 [cite: 37]
        print(f"현재 붓 크기: {brush_size}")
    elif key == ord('-'): # - 키: 붓 크기 감소 [cite: 36]
        brush_size = max(1, brush_size - 1) # 최소 1로 제한 [cite: 37]
        print(f"현재 붓 크기: {brush_size}")

cv.destroyAllWindows()