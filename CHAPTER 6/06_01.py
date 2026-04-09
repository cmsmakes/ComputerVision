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