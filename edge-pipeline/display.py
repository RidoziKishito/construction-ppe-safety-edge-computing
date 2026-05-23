import cv2
import numpy as np


def draw_zones(frame, zones_config):
    """Vẽ các vùng an toàn/nguy hiểm lên màn hình"""
    for zone in zones_config.get("zones", []):
        pts = np.array(zone["polygon"], np.int32).reshape((-1, 1, 2))
        color = (0, 0, 255) if zone["type"] == "danger_zone" else (0, 255, 255)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(
            frame,
            zone["name"],
            (pts[0][0][0], pts[0][0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_person_alert(frame, box, center_point, alert_level, ppe_violations):
    """Vẽ bounding box của người, chấm tâm chân và text cảnh báo"""
    x1, y1, x2, y2 = box

    # Chọn màu vẽ theo Alert Level
    color = (0, 255, 0)  # Xanh lá cho NORMAL
    if alert_level == "WARNING":
        color = (0, 165, 255)  # Cam
    elif alert_level == "CRITICAL":
        color = (0, 0, 255)  # Đỏ

    # Vẽ Box người
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Vẽ tâm chân (chấm tròn nhỏ)
    cv2.circle(frame, center_point, 5, color, -1)

    # Hiển thị text cảnh báo
    label = f"{alert_level} | PPE Miss: {len(ppe_violations)}"
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
