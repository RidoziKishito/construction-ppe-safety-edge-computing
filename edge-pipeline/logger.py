import csv
import os
import cv2
from datetime import datetime


class EdgeLogger:
    def __init__(self, log_dir="logs", filename="violations.csv"):
        self.log_dir = log_dir
        # Tự động tạo thư mục logs nếu chưa có
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Tên file log sẽ gắn ngày tháng để dễ quản lý theo ca làm việc
        self.filepath = os.path.join(
            self.log_dir, f"{datetime.now().strftime('%Y%m%d')}_{filename}"
        )

        # Tạo header nếu file csv chưa tồn tại
        if not os.path.isfile(self.filepath):
            with open(self.filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "camera_id",
                        "frame_id",
                        "zone_id",
                        "alert_level",
                        "violation_type",
                        "confidence",
                        "bbox",
                        "snapshot_path",  # CỘT MỚI: Đường dẫn ảnh chụp
                    ]
                )

    def log_violation(
        self,
        camera_id,
        frame_id,
        active_zones,
        alert_level,
        violations,
        conf,
        bbox,
        frame_img=None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Gộp danh sách zone và lỗi thành chuỗi (ví dụ: "Z01|Z02", "no_helmet|no_vest")
        zone_str = "|".join([z["id"] for z in active_zones]) if active_zones else "None"
        viol_str = "|".join(violations) if violations else "None"
        bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"

        snapshot_path = "None"
        # XỬ LÝ LƯU ẢNH SNAPSHOT
        if frame_img is not None:
            # Tạo thư mục con snapshots
            snap_dir = os.path.join(self.log_dir, "snapshots")
            if not os.path.exists(snap_dir):
                os.makedirs(snap_dir)

            # Đặt tên ảnh theo camera, frame và thời gian
            safe_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            snapshot_filename = f"snap_cam{camera_id}_f{frame_id}_{safe_time}.jpg"
            snapshot_path = os.path.join(snap_dir, snapshot_filename)

            # Ghi file ảnh ra ổ cứng
            cv2.imwrite(snapshot_path, frame_img)

        # Ghi thêm dòng mới vào file csv
        with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    camera_id,
                    frame_id,
                    zone_str,
                    alert_level,
                    viol_str,
                    conf,
                    bbox_str,
                    snapshot_path,
                ]
            )
