from ultralytics import YOLO

model = YOLO("models/signs_best.pt")

def detect_signs(frame):
    results = model(frame, conf=0.4, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf)
            })
    return detections

