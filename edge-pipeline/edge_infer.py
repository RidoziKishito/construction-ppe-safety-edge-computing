import argparse
import cv2
import math
import json
import numpy as np
from ultralytics import YOLO
from logger import EdgeLogger
import rule_engine
import display  # NEW IMPORT TO DRAW UI


# --- TEMPORAL NOISE FILTER (TEMPORAL SMOOTHER) ---
class TemporalSmoother:
    def __init__(self, required_frames=5):
        self.required_frames = required_frames
        # Store state for each Zone: {"Z01": {"level": "CRITICAL", "count": 3}}
        self.zones_status = {}

    def process_alerts(self, current_frame_alerts):
        valid_logs = []
        current_zones = current_frame_alerts.keys()
        all_tracked_zones = set(self.zones_status.keys()).union(set(current_zones))

        for z_id in all_tracked_zones:
            if z_id not in self.zones_status:
                self.zones_status[z_id] = {"level": "NORMAL", "count": 0}

            current_level = current_frame_alerts.get(z_id, {"level": "NORMAL"})["level"]

            # If the alert level matches previous frame, increment counter
            if current_level == self.zones_status[z_id]["level"]:
                self.zones_status[z_id]["count"] += 1
            else:
                # If level changed, reset counter to 1
                self.zones_status[z_id]["level"] = current_level
                self.zones_status[z_id]["count"] = 1

            # Only log when level has been stable for required number of frames
            if self.zones_status[z_id]["level"] in ["WARNING", "CRITICAL"]:
                if self.zones_status[z_id]["count"] >= self.required_frames:
                    valid_logs.append(current_frame_alerts[z_id]["log_data"])

        return valid_logs


# ---------------------------------------------


def load_zones(zones_path):
    try:
        with open(zones_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: zones.json not found, running without zones.")
        return {"zones": []}


def run_pipeline(
    model_path, source, conf_thres, iou_thres, classes_path, zones_path, smooth_frames
):
    model = YOLO(model_path)
    zones_config = load_zones(zones_path)

    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    window_name = "Trinity Edge - Safety Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    safety_logger = EdgeLogger()
    # Initialize smoother with number of frames from CLI argument
    smoother = TemporalSmoother(required_frames=smooth_frames)
    frame_id = 0

    print(
        f"Running video stream. Smoothing = {smooth_frames} frames. Press 'q' or 'X' to exit."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # 1. Call draw_zones from display.py (replaces older code)
        display.draw_zones(frame, zones_config)

        results = model(
            frame, stream=True, verbose=False, conf=conf_thres, iou=iou_thres
        )

        persons = []
        other_detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls_name = model.names[int(box.cls[0])]

                det_obj = {
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "class_name": cls_name,
                }
                if cls_name.lower() == "person":
                    persons.append(det_obj)
                else:
                    other_detections.append(det_obj)

        current_frame_alerts = {}

        for p in persons:
            x1, y1, x2, y2 = p["box"]
            center_point = ((x1 + x2) // 2, y2)

            active_zones = rule_engine.get_person_zones(center_point, zones_config)
            ppe_violations = rule_engine.check_ppe_violation(p["box"], other_detections)
            alert_level = rule_engine.classify_alert(active_zones, ppe_violations)

            zone_str = (
                "|".join([z["id"] for z in active_zones]) if active_zones else "NO_ZONE"
            )
            existing_level = current_frame_alerts.get(zone_str, {}).get(
                "level", "NORMAL"
            )

            # Prefer the most severe alert if multiple people are in the same zone
            if alert_level == "CRITICAL" or (
                alert_level == "WARNING" and existing_level != "CRITICAL"
            ):
                current_frame_alerts[zone_str] = {
                    "level": alert_level,
                    "log_data": {
                        "camera_id": str(source),
                        "frame_id": frame_id,
                        "active_zones": active_zones,
                        "alert_level": alert_level,
                        "violations": ppe_violations,
                        "conf": p["conf"],
                        "bbox": p["box"],
                    },
                }

            # 2. CALL PERSON DRAWING FUNCTION FROM display.py
            display.draw_person_alert(
                frame, p["box"], center_point, alert_level, ppe_violations
            )

        # 3. Push whole-frame state through smoother and then log
        valid_logs = smoother.process_alerts(current_frame_alerts)
        for log_data in valid_logs:
            # PASS CURRENT FRAME FOR SNAPSHOT (image will contain boxes and zones)
            log_data["frame_img"] = frame
            safety_logger.log_violation(**log_data)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) == -1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n.onnx")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--class-names", type=str, default="configs/ppe_classes.txt")
    parser.add_argument("--zones", type=str, default="configs/zones.json")
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Number of consecutive frames required to activate logging",
    )

    args = parser.parse_args()
    run_pipeline(
        args.model,
        args.source,
        args.conf,
        args.iou,
        args.class_names,
        args.zones,
        args.smooth,
    )
