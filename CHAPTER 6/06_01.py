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