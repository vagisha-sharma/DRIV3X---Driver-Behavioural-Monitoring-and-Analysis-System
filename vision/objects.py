#updated objects.py for tailgating
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5,
        verbose=False
    )
    return results[0]  




















# <-----previous objects.py code for future purposes--------->
# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")

# def detect_objects(frame):
#     results = model(frame, conf=0.4, verbose=False)
#     detections = []

#     for r in results:
#         for box in r.boxes:
#             detections.append({
#                 "class": model.names[int(box.cls)],
#                 "bbox": box.xyxy[0].tolist(),
#                 "confidence": float(box.conf)
#             })

#     return detections

