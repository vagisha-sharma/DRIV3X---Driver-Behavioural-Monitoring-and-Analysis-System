class LaneDepartureDetector:
    def __init__(self, offset_threshold=50, cooldown=2.0):
        """
        offset_threshold: pixels from center to consider as lane departure
        cooldown: minimum seconds between repeated events
        """
        self.offset_threshold = offset_threshold
        self.cooldown = cooldown
        self.last_event_time = -1e9  # initialize far in past

    def update(self, offset_px, timestamp):
        """
        offset_px: positive -> vehicle left of lane center, negative -> right
        timestamp: current time in seconds
        """
        if abs(offset_px) < self.offset_threshold:
            return None

        if timestamp - self.last_event_time < self.cooldown:
            return None

        self.last_event_time = timestamp

        direction = "left" if offset_px > 0 else "right"
        severity = "high" if abs(offset_px) > self.offset_threshold * 1.5 else "medium"

        return {
            "time": round(timestamp, 2),
            "event": "lane_departure",
            "direction": direction,
            "severity": severity,
            "offset_px": offset_px
        }
