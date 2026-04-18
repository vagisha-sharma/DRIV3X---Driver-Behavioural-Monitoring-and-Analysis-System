import cv2
import json

from core.pipeline import run_pipeline
from behaviour.tracker import ObjectTracker
from behaviour.tailgating import TailgatingDetector
from behaviour.lead_vehicle import select_lead_vehicle
from behaviour.lane_departure import LaneDepartureDetector
from behaviour.sign_violation import SignViolationDetector

VIDEO_PATH = "testing/test4.mp4"
OUTPUT_EVENTS_JSON = "events.json"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

tracker = ObjectTracker()
tailgating = TailgatingDetector()
lane_departure = LaneDepartureDetector(offset_threshold=50)
sign_violation = SignViolationDetector()

frame_idx = 0
all_events = []

while True:
    ret, frame = cap.read() 
    if not ret:
        break

    timestamp = frame_idx / fps
    h, w = frame.shape[:2]

    # 1️⃣ Perception
    output = run_pipeline(frame)

    # 2️⃣ Tracking
    tracked_objects = tracker.update(output["objects"])

    # 3️⃣ Lead vehicle
    lead_vehicle = select_lead_vehicle(
        tracked_objects,
        output["lanes"]
    )

    # 4️⃣ Tailgating
    tg_event = tailgating.update(lead_vehicle, h, timestamp)
    if tg_event:
        print("⚠️ EVENT:", tg_event)
        all_events.append(tg_event)

    # 5️⃣ Lane departure
    ld_event = lane_departure.update(
        output["lanes"]["offset_px"],
        timestamp
    )
    if ld_event:
        ld_event["offset_px"] = int(ld_event["offset_px"])
        print("⚠️ EVENT:", ld_event)
        all_events.append(ld_event)

    # 6️⃣ Sign violation
    signs = output.get("signs", [])

    if signs:
        for s in signs:
            x1, y1, x2, y2 = s["bbox"]
            ratio = ((x2 - x1) * (y2 - y1)) / (h * w)

            sign_violation.update(
                s["class"],
                ratio,
                s["confidence"]
            )
    else:
        sign_violation.update("none", 0, 0)

    sign_violation.update_states()

    # Placeholder speed
    speed = 0  # replace later

    sv_event = sign_violation.check(speed, timestamp)
    if sv_event:
        print("⚠️ EVENT:", sv_event)
        all_events.append(sv_event)

    frame_idx += 1

# Save all events
with open(OUTPUT_EVENTS_JSON, "w") as f:
    json.dump(all_events, f, indent=2)

print(f"✅ Events saved to {OUTPUT_EVENTS_JSON}")



cap.release()
cv2.destroyAllWindows()
