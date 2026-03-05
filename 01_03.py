import cv2 as cv

# 전역 변수 초기화
drawing = False
start_x, start_y = -1, -1
img = None
temp_img = None
roi = None

def select_roi(event, x, y, flags, param):
    global drawing, start_x, start_y, img, temp_img, roi

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y # 클릭한 시작점 저장 [cite: 59]

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img.copy() # 원본 훼손을 막기 위해 복사본에 그리기
            cv.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2) # 드래그 영역 시각화 [cite: 63]
            cv.imshow('ROI Selection', temp_img)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv.imshow('ROI Selection', temp_img)

        # 시작점과 끝점의 좌표를 정렬 (우하단에서 좌상단으로 드래그할 경우 대비)
        x1, x2 = min(start_x, x), max(start_x, x)
        y1, y2 = min(start_y, y), max(start_y, y)
        
        # 유효한 영역이 선택되었는지 확인 후 ROI 추출 (Numpy 슬라이싱) [cite: 64]
        if x2 - x1 > 0 and y2 - y1 > 0:
            roi = img[y1:y2, x1:x2]
            cv.imshow('ROI', roi) # 잘라낸 영역을 별도의 창에 출력 [cite: 60]

# 1. 이미지 로드 [cite: 57]
img = cv.imread('girl_laughing.jpg') # 본인의 이미지 파일명으로 변경하세요
if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit()
    
temp_img = img.copy()

cv.namedWindow('ROI Selection')
cv.setMouseCallback('ROI Selection', select_roi) # 마우스 콜백 함수 등록 [cite: 58]

print("'r': 영역 선택 리셋, 's': ROI 저장, 'q': 종료")

while True:
    cv.imshow('ROI Selection', temp_img)
    key = cv.waitKey(1) & 0xFF

    if key == ord('r'): # r 키: 영역 선택 리셋 [cite: 61]
        temp_img = img.copy()
        if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow('ROI') # 기존 ROI 창 닫기
    elif key == ord('s'): # s 키: 선택한 영역(ROI) 저장 [cite: 62]
        if roi is not None:
            cv.imwrite('saved_roi.jpg', roi) # 이미지 파일로 저장 [cite: 65]
            print("ROI가 'saved_roi.jpg'로 저장되었습니다.")
        else:
            print("저장할 영역이 선택되지 않았습니다.")
    elif key == ord('q'):
        break

cv.destroyAllWindows()