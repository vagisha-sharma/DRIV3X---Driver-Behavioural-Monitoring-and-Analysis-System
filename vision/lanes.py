import cv2
import numpy as np

def detect_lanes(frame):
    frame = cv2.resize(frame, (640, 480))

    # Perspective transform points (hardcoded for now)
    tl = (222,387)
    bl = (70 ,472)
    tr = (400,380)
    br = (538,472)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(frame, matrix, (640,480))

    hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)

    # FIXED thresholds (no trackbars)
    lower = np.array([0, 0, 200])
    upper = np.array([255, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    y = 472
    left_points, right_points = [], []

    while y > 0:
        # Left window
        left_img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(left_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                left_base = left_base - 50 + cx
                left_points.append((left_base, y))

        # Right window
        right_img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(right_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                right_base = right_base - 50 + cx
                right_points.append((right_base, y))

        y -= 40

    # Vehicle offset (simple estimate)
    lane_center = (left_base + right_base) // 2
    frame_center = frame.shape[1] // 2
    offset_px = frame_center - lane_center

    return {
        "left_lane": left_points,
        "right_lane": right_points,
        "offset_px": offset_px
    }