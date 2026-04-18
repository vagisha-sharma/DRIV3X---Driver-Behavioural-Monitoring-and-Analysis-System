class TailgatingDetector:
    def __init__(self, ratio_threshold=0.25, min_duration=1.5, cooldown=3.0):
        self.ratio_threshold = ratio_threshold
        self.min_duration = min_duration
        self.cooldown = cooldown

        self.history = {}          # track_id -> [(t, ratio)]
        self.last_event_time = {}  # track_id -> time

    def update(self, lead_vehicle, frame_height, timestamp):
        if lead_vehicle is None:
            return None

        tid = lead_vehicle["id"]
        x1, y1, x2, y2 = lead_vehicle["bbox"]

        ratio = (y2 - y1) / frame_height

        self.history.setdefault(tid, []).append((timestamp, ratio))

        # keep only last 3 seconds
        self.history[tid] = [
            (t, r) for (t, r) in self.history[tid]
            if timestamp - t <= 3.0
        ]

        above = [t for (t, r) in self.history[tid] if r > self.ratio_threshold]
        if not above:
            return None

        if max(above) - min(above) < self.min_duration:
            return None

        last = self.last_event_time.get(tid, -1e9)
        if timestamp - last < self.cooldown:
            return None

        self.last_event_time[tid] = timestamp

        return {
            "time": round(timestamp, 2),
            "event": "tailgating",
            "severity": "high" if ratio > 0.30 else "medium",
            "track_id": tid,
            "ratio": round(ratio, 3)
        }
