import argparse
import cv2
import math
import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from logger import EdgeLogger
import rule_engine
import display


class TemporalSmoother:
    def __init__(self, required_frames=5):
        self.required_frames = required_frames
        self.zones_status = {}

    def process_alerts(self, current_frame_alerts):
        valid_logs = []
        current_zones = current_frame_alerts.keys()
        all_tracked_zones = set(self.zones_status.keys()).union(set(current_zones))

        for z_id in all_tracked_zones:
            if z_id not in self.zones_status:
                self.zones_status[z_id] = {"level": "NORMAL", "count": 0}

            current_level = current_frame_alerts.get(z_id, {"level": "NORMAL"})["level"]

            if current_level == self.zones_status[z_id]["level"]:
                self.zones_status[z_id]["count"] += 1
            else:
                self.zones_status[z_id]["level"] = current_level
                self.zones_status[z_id]["count"] = 1

            if self.zones_status[z_id]["level"] in ["WARNING", "CRITICAL"]:
                if self.zones_status[z_id]["count"] >= self.required_frames:
                    valid_logs.append(current_frame_alerts[z_id]["log_data"])

        return valid_logs


def load_zones(zones_path):
    try:
        with open(zones_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: zones.json not found, running without zones.")
        return {"zones": []}


def validate_zone_profile(zones_config, width, height):
    profile_width = zones_config.get("width")
    profile_height = zones_config.get("height")
    if profile_width and profile_height:
        if int(profile_width) != width or int(profile_height) != height:
            raise ValueError(
                "Zone profile resolution "
                f"{profile_width}x{profile_height} does not match video "
                f"resolution {width}x{height}."
            )


def run_pipeline(
    model_path,
    source,
    conf_thres,
    iou_thres,
    classes_path,
    zones_path,
    smooth_frames,
    log_dir="logs",
    output_path=None,
    headless=False,
    max_frames=None,
    camera_id=None,
):
    model = YOLO(model_path)
    zones_config = load_zones(zones_path)

    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    # Lấy thông số video gốc
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    validate_zone_profile(zones_config, w, h)
    out_path = output_path or str(Path(str(source)).with_suffix("")) + "_output.mp4"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    window_name = "Trinity Edge - Safety Pipeline"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    base_source_name = os.path.splitext(os.path.basename(str(source)))[0]
    safety_logger = EdgeLogger(
        log_dir=log_dir,
        source_name=base_source_name,
        filename="events.csv" if Path(log_dir).name != "logs" else None,
    )
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
        if max_frames and frame_id > max_frames:
            break
        display.draw_zones(frame, zones_config)

        results = model(
            frame, stream=True, verbose=True, conf=conf_thres, iou=iou_thres
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

        # 1. EVALUATE DETECTED PERSONS
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

            if alert_level == "CRITICAL" or (
                alert_level == "WARNING" and existing_level != "CRITICAL"
            ):
                current_frame_alerts[zone_str] = {
                    "level": alert_level,
                    "log_data": {
                        "camera_id": camera_id or base_source_name,
                        "frame_id": frame_id,
                        "active_zones": active_zones,
                        "alert_level": alert_level,
                        "violations": ppe_violations,
                        "conf": p["conf"],
                        "bbox": p["box"],
                    },
                }

            display.draw_person_alert(
                frame, p["box"], center_point, alert_level, ppe_violations
            )

        # 2. EVALUATE ALL STANDALONE DETECTIONS (Because Model is missing 'Person')
        violation_classes = [
            "no_helmet",
            "no_goggle",
            "no_gloves",
            "no_boots",
            "no_vest",
            "none",
        ]
        debug_draw_only = []

        for det in other_detections:
            vx_center = (det["box"][0] + det["box"][2]) // 2
            vy_center = (det["box"][1] + det["box"][3]) // 2
            is_inside_person = False

            for p in persons:
                px1, py1, px2, py2 = p["box"]
                if px1 <= vx_center <= px2 and py1 <= vy_center <= py2:
                    is_inside_person = True
                    break

            if not is_inside_person:
                # Dùng chính vật thể này (helmet, no_helmet) làm mỏ neo để check Zone
                center_point = (vx_center, det["box"][3])
                active_zones = rule_engine.get_person_zones(center_point, zones_config)

                cls_name_lower = det["class_name"].lower()
                ppe_violations = (
                    [det["class_name"]] if cls_name_lower in violation_classes else []
                )

                # Mang mỏ neo này đi hỏi Rule Engine
                alert_level = rule_engine.classify_alert(active_zones, ppe_violations)

                if alert_level != "NORMAL":
                    zone_str = (
                        "|".join([z["id"] for z in active_zones])
                        if active_zones
                        else "NO_ZONE"
                    )
                    existing_level = current_frame_alerts.get(zone_str, {}).get(
                        "level", "NORMAL"
                    )

                    if alert_level == "CRITICAL" or (
                        alert_level == "WARNING" and existing_level != "CRITICAL"
                    ):
                        current_frame_alerts[zone_str] = {
                            "level": alert_level,
                            "log_data": {
                                "camera_id": camera_id or base_source_name,
                                "frame_id": frame_id,
                                "active_zones": active_zones,
                                "alert_level": alert_level,
                                "violations": ppe_violations,
                                "conf": det["conf"],
                                "bbox": det["box"],
                            },
                        }
                    # Nếu có Alert (Warning/Critical) thì vẽ khung báo động
                    display.draw_person_alert(
                        frame, det["box"], center_point, alert_level, ppe_violations
                    )
                else:
                    # Nếu NORMAL (ví dụ đội nón đứng ngoài Zone), thì chỉ vẽ khung xanh lơ
                    debug_draw_only.append(det)

        # 3. DRAW NORMAL PPEs FOR DEBUGGING
        display.draw_other_detections(frame, debug_draw_only)

        # 4. FILTER ALERTS & LOG
        valid_logs = smoother.process_alerts(current_frame_alerts)
        for log_data in valid_logs:
            log_data["frame_img"] = frame
            safety_logger.log_violation(**log_data)

        writer.write(frame)
        if not headless:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) == -1:
                    break
            except cv2.error:
                break

    cap.release()
    writer.release()
    print(f"Output video saved to: {out_path}")
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best.onnx")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--class-names", type=str, default="configs/ppe_classes.txt")
    parser.add_argument("--zones", type=str, default="configs/zones.json")
    parser.add_argument("--smooth", type=int, default=5, help="Smooth frames")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--camera-id")

    args = parser.parse_args()
    run_pipeline(
        args.model,
        args.source,
        args.conf,
        args.iou,
        args.class_names,
        args.zones,
        args.smooth,
        args.log_dir,
        args.output,
        args.headless,
        args.max_frames,
        args.camera_id,
    )
