import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description="Trích xuất frame đầu tiên của video để lấy độ phân giải.")
    parser.add_argument("--source", type=str, required=True, help="Đường dẫn đến file video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video: {args.source}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame đầu tiên từ video.")
        cap.release()
        return

    print("Resolution:", frame.shape[1], "x", frame.shape[0])  
    
    import os
    base_name = os.path.splitext(os.path.basename(args.source))[0]
    out_filename = f"frame_check_{base_name}.jpg"
    
    cv2.imwrite(out_filename, frame)
    print(f"Đã lưu frame đầu tiên thành '{out_filename}'")
    
    cap.release()
    
    # Mở giao diện Zone Editor
    import subprocess
    print("Đang mở công cụ chọn Zone...")
    subprocess.run(["python", "zone_editor.py", out_filename])

if __name__ == "__main__":
    main()