def select_lead_vehicle(tracked_objects, lanes):
    if not lanes:
        return None

    left_lane = lanes.get("left_lane")
    right_lane = lanes.get("right_lane")

    if not left_lane or not right_lane:
        return None

    left_x = left_lane[0][0]
    right_x = right_lane[0][0]

    candidates = []

    for obj in tracked_objects:
        if obj["class_id"] not in (2, 5, 7):  # car, bus, truck
            continue

        x1, y1, x2, y2 = obj["bbox"]
        cx = (x1 + x2) / 2

        if left_x < cx < right_x:
            candidates.append(obj)

    if not candidates:
        return None

    # closest vehicle = largest bbox height
    return max(candidates, key=lambda o: o["bbox"][3] - o["bbox"][1])
