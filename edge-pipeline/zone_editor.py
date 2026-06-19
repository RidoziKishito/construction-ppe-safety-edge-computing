import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk


class ZoneEditor:
    def __init__(
        self,
        root,
        image_path,
        output_path,
        source_name=None,
        source_width=None,
        source_height=None,
    ):
        self.root = root
        self.root.title("Trinity Zone Editor")
        self.output_path = Path(output_path)
        self.source_name = source_name
        self.source_width = source_width
        self.source_height = source_height

        self.original_image = Image.open(image_path)
        self.display_image, self.scale_factor = self._fit_image(self.original_image)
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        self.current_mode = None
        self.first_point = None
        self.temp_marker = None
        self.points = {"danger": [], "warning": []}
        self.rectangles = {"danger": None, "warning": None}
        self.colors = {"danger": "red", "warning": "gold"}

        self._build_ui()

    @staticmethod
    def _fit_image(image):
        max_width, max_height = 1000, 650
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        if scale == 1.0:
            return image.copy(), scale
        size = (int(image.width * scale), int(image.height * scale))
        return image.resize(size, Image.Resampling.LANCZOS), scale

    def _build_ui(self):
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.display_image.width,
            height=self.display_image.height,
            cursor="cross",
        )
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.bind("<Button-1>", self.on_click)

        controls = tk.Frame(self.root)
        controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.status = tk.Label(
            controls,
            text="Select a zone type, then click two opposite corners.",
            fg="navy",
            wraplength=180,
        )
        self.status.pack(pady=10)

        tk.Button(
            controls,
            text="Danger Zone",
            bg="red",
            fg="white",
            width=16,
            command=lambda: self.set_mode("danger"),
        ).pack(pady=5)
        tk.Button(
            controls,
            text="Warning Zone",
            bg="gold",
            fg="black",
            width=16,
            command=lambda: self.set_mode("warning"),
        ).pack(pady=5)
        tk.Button(
            controls,
            text="Save",
            bg="green",
            fg="white",
            width=16,
            command=self.save_and_exit,
        ).pack(side=tk.BOTTOM, pady=20)

    def set_mode(self, mode):
        self.current_mode = mode
        self.first_point = None
        self.points[mode] = []
        if self.rectangles[mode]:
            self.canvas.delete(self.rectangles[mode])
            self.rectangles[mode] = None
        self.status.config(
            text=f"Drawing {mode} zone: click two opposite corners.",
            fg=self.colors[mode],
        )

    def on_click(self, event):
        if not self.current_mode:
            messagebox.showwarning("Select zone", "Select a zone type before drawing.")
            return

        if self.first_point is None:
            self.first_point = (event.x, event.y)
            self.temp_marker = self.canvas.create_oval(
                event.x - 3,
                event.y - 3,
                event.x + 3,
                event.y + 3,
                fill=self.colors[self.current_mode],
            )
            return

        x1, y1 = self.first_point
        x2, y2 = event.x, event.y
        self.canvas.delete(self.temp_marker)
        bounds = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        self.points[self.current_mode] = bounds
        self.rectangles[self.current_mode] = self.canvas.create_rectangle(
            *bounds,
            outline=self.colors[self.current_mode],
            width=3,
        )
        self.status.config(text=f"{self.current_mode.title()} zone ready.", fg="green")
        self.current_mode = None
        self.first_point = None

    def _zone_from_bounds(self, mode, zone_id, name, zone_type):
        bounds = self.points[mode]
        if not bounds:
            return None
        x1, y1, x2, y2 = (int(value / self.scale_factor) for value in bounds)
        return {
            "id": zone_id,
            "name": name,
            "type": zone_type,
            "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        }

    def save_and_exit(self):
        new_zones = [
            zone
            for zone in (
                self._zone_from_bounds("danger", "Z01", "Danger Zone", "danger_zone"),
                self._zone_from_bounds("warning", "Z02", "Warning Zone", "warning_zone"),
            )
            if zone
        ]
        if not new_zones:
            messagebox.showinfo("No changes", "No zones were drawn.")
            return

        try:
            existing = json.loads(self.output_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {"zones": []}

        updated_types = {zone["type"] for zone in new_zones}
        zones = [
            zone
            for zone in existing.get("zones", [])
            if zone.get("type") not in updated_types
        ]
        zones.extend(new_zones)

        data = {
            "source": self.source_name,
            "width": self.source_width or self.original_image.width,
            "height": self.source_height or self.original_image.height,
            "zones": zones,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        messagebox.showinfo("Saved", f"Zone profile saved to:\n{self.output_path}")
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Create safety zones on a video frame.")
    parser.add_argument("image_path")
    parser.add_argument("--output", default="configs/zones.json")
    parser.add_argument("--source")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()

    root = tk.Tk()
    ZoneEditor(
        root,
        args.image_path,
        args.output,
        args.source,
        args.width,
        args.height,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
