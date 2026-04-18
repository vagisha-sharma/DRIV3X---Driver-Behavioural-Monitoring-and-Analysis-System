class ObjectTracker:
    def __init__(self):
        pass

    def update(self, yolo_results):
        """
        yolo_results: Ultralytics YOLO Results object (model(frame)[0])
        """

        tracked = []

        if yolo_results.boxes is None:
            return tracked

        boxes = yolo_results.boxes

        for i in range(len(boxes)):
            box = boxes[i]

            if box.id is None:
                continue  # not tracked yet

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tracked.append({
                "id": int(box.id.item()),
                "class_id": int(box.cls.item()),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf.item())
            })

        return tracked
