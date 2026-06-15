import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import cv2
import threading
import csv
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import easyocr
import logging
import winsound
import re

logging.getLogger("ultralytics").setLevel(logging.WARNING)

PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')

COLOR_BG_MAIN = "#0f172a"
COLOR_BG_PANEL = "#1e293b"
COLOR_ACCENT = "#06b6d4"
COLOR_TEXT_DIM = "#94a3b8"
COLOR_SUCCESS = "#10b981"
COLOR_DANGER = "#ef4444"

class TrafficDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Guardian Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLOR_BG_MAIN)
        self.root.state('zoomed')

        print("Initialising AI models... please wait.")
        self.model = YOLO('yolov8n.pt')
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("AI Ready!")

        self.running = False
        self.cap = None
        self.camera_index = 0
        self.detected_objects = []
        self.next_object_id = 0

        self.is_ocr_busy = False
        self.frame_count = 0
        self.skip_frames = 3
        self.total_violations = 0

        self.vehicle_history = {} 
        self.violation_counter = {} 
        self.captured_ids = set() 
        self.FRAME_MEMORY = 12

        self.db_file = 'traffic_data.csv'
        self.db_lock = threading.Lock() 
        print(f"SAVING DATA TO: {os.path.abspath(self.db_file)}")
        
        self.setup_database()
        self.create_layout()

    def setup_database(self):
        if not os.path.isfile(self.db_file):
            with open(self.db_file, mode='w', newline='') as f:
                csv.writer(f).writerow(['Timestamp', 'Violation', 'Plate', 'Evidence_File', 'Total_Offenses'])

    def check_blacklist_and_save(self, plate_text, violation_type, image_crop):
        with self.db_lock:
            try:
                rows = []
                offense_count = 1
                is_repeat = False

                if os.path.isfile(self.db_file):
                    with open(self.db_file, mode='r', newline='') as f:
                        rows = list(csv.reader(f))

                if plate_text not in ("UNREADABLE", "ERROR"):
                    for i in range(1, len(rows)):
                        if len(rows[i]) >= 5 and rows[i][2] == plate_text:
                            offense_count = int(rows[i][4]) + 1
                            rows[i][4] = str(offense_count)
                            is_repeat = True
                            break

                now = datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                filename = f"evidence_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"

                if not cv2.imwrite(filename, image_crop):
                    print(f"[WARN] Could not save evidence image: {filename}")

                rows.append([time_str, violation_type, plate_text, filename, str(offense_count)])

                with open(self.db_file, mode='w', newline='') as f:
                    csv.writer(f).writerows(rows)

                self.total_violations += 1

                if is_repeat:
                    threading.Thread(target=lambda: (winsound.Beep(2000,400), winsound.Beep(2000,400)), daemon=True).start()
                    self.root.after(0, lambda p=plate_text, c=offense_count: self.log_message(f"🚨 REPEAT OFFENDER: {p}  ({c}x)", True))
                else:
                    threading.Thread(target=lambda: winsound.Beep(1000, 500), daemon=True).start()
                    self.root.after(0, lambda p=plate_text: self.log_message(f"⚠  RECORDED: {p}"))

                self.root.after(0, lambda: self.lbl_violation_count.config(text=str(self.total_violations)))
                self.root.after(0, lambda fn=filename: self.update_evidence_preview(fn))

            except Exception as e:
                print(f"[DB ERROR] {e}")

    def ocr_worker(self, image, violation_type):
        try:
            h, w = image.shape[:2]
            plate_strip = image[int(h * 0.55): h, :]
            big = cv2.resize(plate_strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 11, 17, 17)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blur)

            _, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_adapt = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            ALLOW = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            res1 = self.reader.readtext(thresh_otsu, allowlist=ALLOW, mag_ratio=1.5, detail=1)
            res2 = self.reader.readtext(thresh_adapt, allowlist=ALLOW, mag_ratio=1.5, detail=1)

            def best_plate(results):
                tokens = [r[1].replace(" ", "").upper() for r in results if r[2] >= 0.4]
                return "".join(tokens)

            plate1 = best_plate(res1)
            plate2 = best_plate(res2)
            plate = plate1 if len(plate1) >= len(plate2) else plate2

            clean = re.sub(r'[^A-Z0-9]', '', plate.upper())
            if PLATE_PATTERN.match(clean):
                final_plate = clean
            elif len(clean) >= 4:
                final_plate = clean
            else:
                final_plate = "UNREADABLE"

            self.check_blacklist_and_save(final_plate, violation_type, image)

        except Exception as e:
            print(f"[OCR ERROR] {e}")
            self.check_blacklist_and_save("ERROR", violation_type, image)
        finally:
            self.is_ocr_busy = False

    def score_wrong_way(self, obj_id, cx, bottom_y, box_width, frame_w):
        history = self.vehicle_history.get(obj_id, [])
        if len(history) < 8:
            return False, 0

        old_cx, old_by, old_bw = history[-8]
        delta_y = bottom_y - old_by
        growth_rate = (box_width - old_bw) / max(old_bw, 1)

        score = 0
        if delta_y > 35: score += 3
        elif delta_y > 20: score += 1
        
        if growth_rate > 0.15: score += 3
        elif growth_rate > 0.08: score += 1
        
        if cx > frame_w * 0.35: score += 1

        return score >= 5, score

    def update_frame(self):
        if not self.running:
            return

        success, frame = self.cap.read()
        if not success:
            self.stop_system()
            return

        clean_frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        self.frame_count += 1

        if self.frame_count % self.skip_frames == 0:
            results = self.model(frame, stream=True, verbose=False, conf=0.45)
            new_dets = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) not in [2, 3, 5, 7]:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    bw = x2 - x1
                    new_dets.append({'box': (x1,y1,x2,y2), 'center_x': cx, 'bottom_y': y2, 'box_width': bw})

            matched_objs = []
            remaining = list(self.detected_objects)
            for nd in new_dets:
                best_match, best_dist = None, 120
                for ko in remaining:
                    dist = (abs(nd['center_x'] - ko['center_x']) + abs(nd['bottom_y'] - ko['bottom_y'])) / 2
                    if dist < best_dist:
                        best_match, best_dist = ko, dist
                if best_match:
                    matched_objs.append({**nd, 'id': best_match['id']})
                    remaining.remove(best_match)
                else:
                    matched_objs.append({**nd, 'id': self.next_object_id})
                    self.next_object_id += 1
            self.detected_objects = matched_objs

        zone_y = int(frame_h * 0.70)
        cv2.line(frame, (0, zone_y), (frame_w, zone_y), (0, 0, 255), 2)
        cv2.putText(frame, "CAPTURE ZONE", (20, zone_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            obj_id = obj['id']
            cx = obj['center_x']
            bottom_y = obj['bottom_y']
            bw = obj['box_width']

            if obj_id not in self.vehicle_history:
                self.vehicle_history[obj_id] = []
            self.vehicle_history[obj_id].append((cx, bottom_y, bw))
            if len(self.vehicle_history[obj_id]) > self.FRAME_MEMORY:
                self.vehicle_history[obj_id].pop(0)

            is_violation, score = self.score_wrong_way(obj_id, cx, bottom_y, bw, frame_w)

            if is_violation:
                self.violation_counter[obj_id] = self.violation_counter.get(obj_id, 0) + 1
                color = (0, 0, 255)
                label = f"WRONG WAY [{score}/7]"
            else:
                self.violation_counter[obj_id] = 0
                color = (0, 255, 0)
                label = "SAFE"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            consec = self.violation_counter.get(obj_id, 0)
            if (is_violation and consec >= 4 and bottom_y > zone_y and bw > 120 and obj_id not in self.captured_ids and not self.is_ocr_busy):
                self.is_ocr_busy = True
                self.captured_ids.add(obj_id)
                pad = 30
                cy1 = max(0, y1 - pad); cy2 = min(frame_h, y2 + pad)
                cx1_ = max(0, x1 - pad); cx2_ = min(frame_w, x2 + pad)
                roi = clean_frame[cy1:cy2, cx1_:cx2_].copy()

                threading.Thread(target=self.ocr_worker, args=(roi, "Wrong Way"), daemon=True).start()
                self.root.after(0, lambda id=obj_id: self.log_message(f"🎯 Capture fired: vehicle ID {id}"))

        frame_small = cv2.resize(frame, (960, 540))
        img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")
        self.root.after(30, self.update_frame)

    def create_layout(self):
        header = tk.Frame(self.root, bg=COLOR_BG_PANEL, height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="TRAFFIC GUARDIAN", font=("Segoe UI", 24, "bold"), bg=COLOR_BG_PANEL, fg=COLOR_ACCENT).pack(side=tk.LEFT, padx=30)
        
        container = tk.Frame(self.root, bg=COLOR_BG_MAIN)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        vid_section = tk.Frame(container, bg=COLOR_BG_MAIN)
        vid_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vid_border = tk.Frame(vid_section, bg=COLOR_ACCENT, padx=2, pady=2)
        vid_border.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(vid_border, text="CAMERA FEED OFFLINE", bg="black", fg="#334155", font=("Consolas", 20))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(container, bg=COLOR_BG_PANEL, width=400)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(20,0))
        sidebar.pack_propagate(False)

        self._make_card(sidebar, "SYSTEM STATUS", self._status_content)
        self._make_card(sidebar, "SESSION STATISTICS", self._stats_content)
        self._make_card(sidebar, "CONTROLS", self._controls_content)

        tk.Label(sidebar, text="LATEST CAPTURE", bg=COLOR_BG_PANEL, fg=COLOR_ACCENT, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=20, pady=(10,5))
        self.evidence_label = tk.Label(sidebar, bg="black", text="[NO EVIDENCE YET]", fg=COLOR_TEXT_DIM, font=("Segoe UI", 10), height=10)
        self.evidence_label.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(sidebar, text="LIVE EVENT LOG", bg=COLOR_BG_PANEL, fg=COLOR_TEXT_DIM, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=20, pady=(10,5))
        self.log_box = scrolledtext.ScrolledText(sidebar, height=12, bg="#000000", fg="#00ff00", font=("Consolas", 9), bd=0)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,20))

    def _make_card(self, parent, title, content_fn):
        card = tk.Frame(parent, bg="#273548", padx=15, pady=15)
        card.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(card, text=title, bg="#273548", fg=COLOR_ACCENT, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0,10))
        content_fn(card)

    def _status_content(self, p):
        self.status_indicator = tk.Label(p, text="● OFFLINE", bg="#273548", fg=COLOR_DANGER, font=("Segoe UI", 14, "bold"))
        self.status_indicator.pack(anchor="w")

    def _stats_content(self, p):
        f = tk.Frame(p, bg="#273548")
        f.pack(fill=tk.X)
        self.lbl_violation_count = tk.Label(f, text="0", bg="#273548", fg="white", font=("Segoe UI", 28, "bold"))
        self.lbl_violation_count.pack(side=tk.LEFT)
        tk.Label(f, text="Violations\nDetected", bg="#273548", fg=COLOR_TEXT_DIM, font=("Segoe UI", 10), justify=tk.LEFT).pack(side=tk.LEFT, padx=10)

    def _controls_content(self, p):
        s = {"font": ("Segoe UI", 11, "bold"), "bd": 0, "cursor": "hand2", "height": 1}
        self.btn_start = tk.Button(p, text="▶ START MONITORING", bg=COLOR_SUCCESS, fg="white", command=self.start_system, **s)
        self.btn_start.pack(fill=tk.X, pady=5)
        self.btn_stop = tk.Button(p, text="⏹ STOP SYSTEM", bg="#475569", fg="white", command=self.stop_system, state=tk.DISABLED, **s)
        self.btn_stop.pack(fill=tk.X, pady=5)

    def log_message(self, msg, warn=False):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_box.see(tk.END)
        if warn:
            self.log_box.tag_add("w", "end-2l", "end-1c")
            self.log_box.tag_config("w", foreground="red")

    def update_evidence_preview(self, path):
        try:
            img = Image.open(path).resize((360, 200), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            self.evidence_label.config(image=imgtk, height=200)
            self.evidence_label.image = imgtk
        except Exception:
            pass

    def start_system(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(3, 1280); self.cap.set(4, 720)
            self.btn_start.config(state=tk.DISABLED, bg="#475569")
            self.btn_stop.config(state=tk.NORMAL, bg=COLOR_DANGER)
            self.status_indicator.config(text="● ONLINE", fg=COLOR_SUCCESS)
            self.log_message("System Online. AI Active.")
            self.update_frame()

    def stop_system(self):
        self.running = False
        if self.cap: self.cap.release()
        self.video_label.configure(image='', text="CAMERA FEED OFFLINE")
        self.btn_start.config(state=tk.NORMAL, bg=COLOR_SUCCESS)
        self.btn_stop.config(state=tk.DISABLED, bg="#475569")
        self.status_indicator.config(text="● OFFLINE", fg=COLOR_DANGER)
        self.log_message("System Offline.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficDashboard(root)
    root.mainloop()