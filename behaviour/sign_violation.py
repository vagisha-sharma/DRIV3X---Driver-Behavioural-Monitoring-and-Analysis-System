# behaviour/sign_violation.py

class SignViolationDetector:
    def __init__(self, frames_to_check=5, ratio_threshold=0.003):
        self.frames_to_check = frames_to_check
        self.threshold = ratio_threshold

        self.history = [["none", 0, 0]] * frames_to_check

        self.in_red = False
        self.recent_red_conf = 0
        self.in_stop = False
        self.curr_speed_lim = 50

    def update(self, class_name, ratio, confidence):
        self.history.append([class_name, ratio, confidence])
        if len(self.history) > self.frames_to_check:
            self.history.pop(0)

    def _seen(self, label):
        return any(
            c == label and r >= self.threshold
            for c, r, _ in self.history
        )

    def _confidence(self, label):
        return max(
            (conf for c, r, conf in self.history if c == label),
            default=0
        )

    def update_states(self):
        if self._seen("Red Light"):
            self.in_red = True
            self.recent_red_conf = self._confidence("Red Light")

        if self._seen("Green Light"):
            self.in_red = False

        if self._seen("Stop"):
            self.in_stop = True

        speed_limits = {
            f"Speed Limit {i}": i
            for i in [10,20,30,40,50,60,70,80,90,100,110,120]
        }

        for c, r, _ in self.history:
            if c in speed_limits and r >= self.threshold:
                self.curr_speed_lim = speed_limits[c]

    def check(self, speed, timestamp):
        event = None

        # STOP sign violation
        if self.in_stop and speed > 5:
            event = {
                "time": round(timestamp, 2),
                "event": "stop_sign_violation",
                "severity": "high",
                "confidence": self._confidence("Stop")
            }
            self.in_stop = False

        # SPEED limit violation
        elif speed > self.curr_speed_lim:
            event = {
                "time": round(timestamp, 2),
                "event": "speed_limit_exceeded",
                "severity": "medium" if speed < self.curr_speed_lim + 20 else "high",
                "speed": speed,
                "speed_limit": self.curr_speed_lim
            }

        # RED light violation
        elif self.in_red and not self._seen("Red Light"):
            event = {
                "time": round(timestamp, 2),
                "event": "red_light_violation",
                "severity": "high",
                "confidence": self.recent_red_conf
            }
            self.in_red = False

        return event
