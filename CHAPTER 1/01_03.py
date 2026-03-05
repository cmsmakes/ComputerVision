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