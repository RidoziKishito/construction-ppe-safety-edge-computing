import cv2
import numpy as np


def point_in_polygon(point, polygon):
    """
    Kiểm tra xem một điểm (x, y) có nằm trong đa giác không.
    Sử dụng cv2.pointPolygonTest cho tốc độ cực nhanh.
    """
    poly_array = np.array(polygon, np.int32)
    # Trả về > 0 nếu nằm trong, = 0 nếu nằm trên cạnh, < 0 nếu nằm ngoài
    result = cv2.pointPolygonTest(poly_array, point, False)
    return result >= 0


def get_person_zones(person_center, zones_config):
    """
    Trả về danh sách các zone mà người này đang đứng vào.
    """
    active_zones = []
    for zone in zones_config.get("zones", []):
        if point_in_polygon(person_center, zone["polygon"]):
            active_zones.append(zone)
    return active_zones


def check_ppe_violation(person_box, all_detections):
    """
    Kiểm tra xem person này có bị thiếu PPE không.
    Cách làm: Quét các detection thuộc loại 'no_helmet', 'no_vest', v.v.
    Nếu tâm của chúng nằm bên trong bounding box của person -> người đó vi phạm.
    """
    px1, py1, px2, py2 = person_box
    violations = []

    # Danh sách các class vi phạm theo dataset 3.3
    violation_classes = ["no_helmet", "no_vest", "no_gloves", "no_boots", "no_goggle"]

    for det in all_detections:
        cls_name = det["class_name"]
        if cls_name in violation_classes:
            # Lấy tâm của vật thể vi phạm
            vx1, vy1, vx2, vy2 = det["box"]
            v_center_x = (vx1 + vx2) // 2
            v_center_y = (vy1 + vy2) // 2

            # Nếu tâm của vi phạm nằm trong hộp của người này
            if px1 <= v_center_x <= px2 and py1 <= v_center_y <= py2:
                violations.append(cls_name)

    return violations


def classify_alert(active_zones, ppe_violations):
    """
    Phân loại mức độ nguy hiểm dựa trên Zone và PPE.
    Trả về: NORMAL, WARNING, hoặc CRITICAL
    """
    is_danger_zone = any(z["type"] == "danger_zone" for z in active_zones)
    is_warning_zone = any(z["type"] == "warning_zone" for z in active_zones)
    has_violation = len(ppe_violations) > 0

    # Rule 1: Vào vùng Danger -> Báo động đỏ ngay lập tức (bất kể PPE)
    if is_danger_zone:
        return "CRITICAL"

    # Rule 2: Vào vùng Warning + Thiếu PPE -> Báo động đỏ
    if is_warning_zone and has_violation:
        return "CRITICAL"

    # Rule 3: Vào vùng Warning (nhưng đủ PPE) -> Chỉ cảnh báo nhẹ
    if is_warning_zone and not has_violation:
        return "WARNING"

    # Rule 4: Ở vùng an toàn nhưng thiếu PPE -> Cảnh báo nhắc nhở
    if not is_warning_zone and not is_danger_zone and has_violation:
        return "WARNING"

    # Còn lại là bình thường
    return "NORMAL"
