import sys
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class ZoneEditor:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Trinity Zone Editor")
        
        self.image_path = image_path
        self.orig_img = Image.open(image_path)
        
        max_w, max_h = 1000, 650
        self.scale_factor = 1.0
        
        if self.orig_img.width > max_w or self.orig_img.height > max_h:
            scale_w = max_w / self.orig_img.width
            scale_h = max_h / self.orig_img.height
            self.scale_factor = min(scale_w, scale_h)
            new_w = int(self.orig_img.width * self.scale_factor)
            new_h = int(self.orig_img.height * self.scale_factor)
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.ANTIALIAS
            self.display_img = self.orig_img.resize((new_w, new_h), resample_filter)
        else:
            self.display_img = self.orig_img.copy()

        self.tk_img = ImageTk.PhotoImage(self.display_img)
        
        # UI Layout
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=self.display_img.width, height=self.display_img.height, cursor="cross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.lbl_status = tk.Label(self.btn_frame, text="Vui lòng chọn Zone\nđể bắt đầu vẽ.", fg="blue")
        self.lbl_status.pack(pady=10)
        
        self.btn_danger = tk.Button(self.btn_frame, text="Danger Zone", bg="red", fg="white", width=15, command=self.set_mode_danger)
        self.btn_danger.pack(pady=5)
        
        self.btn_warning = tk.Button(self.btn_frame, text="Warning Zone", bg="gold", fg="black", width=15, command=self.set_mode_warning)
        self.btn_warning.pack(pady=5)
        
        self.btn_ok = tk.Button(self.btn_frame, text="OK", bg="green", fg="white", width=15, command=self.save_and_exit)
        self.btn_ok.pack(side=tk.BOTTOM, pady=20)
        
        self.canvas.bind("<Button-1>", self.on_click)
        
        # State
        self.current_mode = None
        self.points = {"danger": [], "warning": []}
        self.rects = {"danger": None, "warning": None}
        self.colors = {"danger": "red", "warning": "yellow"}
        self.temp_point = None

    def set_mode_danger(self):
        self.current_mode = "danger"
        self.temp_point = None
        self.points["danger"] = []
        if self.rects["danger"]:
            self.canvas.delete(self.rects["danger"])
            self.rects["danger"] = None
        self.lbl_status.config(text="Đang vẽ: Danger Zone\nClick 2 điểm chéo nhau", fg="red")

    def set_mode_warning(self):
        self.current_mode = "warning"
        self.temp_point = None
        self.points["warning"] = []
        if self.rects["warning"]:
            self.canvas.delete(self.rects["warning"])
            self.rects["warning"] = None
        self.lbl_status.config(text="Đang vẽ: Warning Zone\nClick 2 điểm chéo nhau", fg="orange")

    def on_click(self, event):
        if not self.current_mode:
            messagebox.showwarning("Cảnh báo", "Hãy click vào nút 'Danger Zone' hoặc 'Warning Zone' trước khi vẽ.")
            return
            
        x, y = event.x, event.y
        if not self.temp_point:
            self.temp_point = (x, y)
            # Draw a temporary small circle
            self.temp_id = self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=self.colors[self.current_mode])
        else:
            x1, y1 = self.temp_point
            x2, y2 = x, y
            self.canvas.delete(self.temp_id)
            
            # Ensure proper ordering top-left to bottom-right
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            self.points[self.current_mode] = [min_x, min_y, max_x, max_y]
            
            rect_id = self.canvas.create_rectangle(min_x, min_y, max_x, max_y, outline=self.colors[self.current_mode], width=3)
            self.rects[self.current_mode] = rect_id
            
            self.lbl_status.config(text=f"Đã vẽ xong {self.current_mode.capitalize()} Zone!", fg="green")
            self.current_mode = None
            self.temp_point = None

    def save_and_exit(self):
        if not self.points["danger"] and not self.points["warning"]:
            messagebox.showinfo("Thông báo", "Không có zone nào được vẽ. Đóng ứng dụng.")
            self.root.destroy()
            return

        zones_data = {"zones": []}
        
        # Danger Zone logic
        if self.points["danger"]:
            x1, y1, x2, y2 = self.points["danger"]
            x1, y1 = int(x1 / self.scale_factor), int(y1 / self.scale_factor)
            x2, y2 = int(x2 / self.scale_factor), int(y2 / self.scale_factor)
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            zones_data["zones"].append({
                "id": "Z01",
                "name": "Danger Zone",
                "type": "danger_zone",
                "polygon": polygon
            })
            
        # Warning Zone logic
        if self.points["warning"]:
            x1, y1, x2, y2 = self.points["warning"]
            x1, y1 = int(x1 / self.scale_factor), int(y1 / self.scale_factor)
            x2, y2 = int(x2 / self.scale_factor), int(y2 / self.scale_factor)
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            zones_data["zones"].append({
                "id": "Z02",
                "name": "Warning Zone",
                "type": "warning_zone",
                "polygon": polygon
            })

        # Read existing to not overwrite un-drawn zones
        try:
            with open("configs/zones.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {"zones": []}

        final_zones = []
        updated_types = [z["type"] for z in zones_data["zones"]]
        for ez in existing_data.get("zones", []):
            if ez["type"] not in updated_types:
                final_zones.append(ez)
        
        final_zones.extend(zones_data["zones"])

        with open("configs/zones.json", "w", encoding="utf-8") as f:
            json.dump({"zones": final_zones}, f, indent=2)
            
        messagebox.showinfo("Thành công", "Đã cập nhật zones.json thành công!\nBây giờ bạn có thể tiếp tục chạy edge_infer.py")
        self.root.destroy()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python zone_editor.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    root = tk.Tk()
    app = ZoneEditor(root, img_path)
    root.mainloop()
