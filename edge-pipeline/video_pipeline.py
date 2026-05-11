import cv2
import math
from ultralytics import YOLO

# Load ONNX model (temporarily using the standard YOLO11n model)
model = YOLO("yolo11n.onnx")


def run_pipeline(video_source):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open the video source or camera.")
        return

    print("Running video stream. Press 'q' or 'X' to exit.")

    window_name = "Trinity Edge - Safety Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or lost connection.")
            break

        # 1. Run inference with the ONNX model
        # stream=True helps optimize memory usage during video processing
        results = model(frame, stream=True, verbose=False)

        # 2. Process results and draw them on the frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates (x_min, y_min, x_max, y_max)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence score
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Get class ID (for this test file, class 0 is Person)
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # For now, only focus on 'person' for the demo
                if class_name == "person":
                    # Draw a rectangle around the person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw the label text
                    label = f"{class_name} {conf}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # TODO: Insert zone checking code here next week

        # Display the frame
        cv2.imshow(window_name, frame)

        # Handle app exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) == -1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Use 0 for the webcam, then move around to see whether it detects a person
    run_pipeline(0)
