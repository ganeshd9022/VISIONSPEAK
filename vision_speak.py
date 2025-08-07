import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import torch
import threading
import queue
import time
import subprocess

class MacObjectDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("macOS Object Detector")
        self.root.geometry("1200x800")
        
        # Detection components
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.announced_objects = set()
        
        # Thread communication
        self.frame_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue()
        self.voice_queue = queue.Queue()
        
        # GUI setup
        self.create_gui()
        self.setup_threads()
        self.voice_enabled = self.check_voice_system()

    def create_gui(self):
        # Video display
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Controls
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_detection)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        
        # Confidence control
        ttk.Label(control_frame, text="Confidence Threshold:").pack()
        self.confidence = ttk.Scale(control_frame, from_=0.1, to=1.0, value=0.4)
        self.confidence.pack()
        
        # Voice control
        self.voice_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Enable Voice", variable=self.voice_var).pack(pady=5)
        
        # Detection log
        log_frame = ttk.LabelFrame(control_frame, text="Detection Log")
        log_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=15, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_threads(self):
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.voice_thread = threading.Thread(target=self.process_voice, daemon=True)

    def check_voice_system(self):
        try:
            subprocess.check_call(['which', 'say'], stdout=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            messagebox.showwarning("Voice Disabled", "macOS 'say' command not found!")
            return False

    def start_detection(self):
        if self.voice_var.get() and not self.voice_enabled:
            messagebox.showwarning("Voice Disabled", "System voice unavailable")
            self.voice_var.set(False)
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.announced_objects.clear()
        
        self.capture_thread.start()
        self.process_thread.start()
        self.voice_thread.start()
        self.update_gui()

    def stop_detection(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def capture_frames(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, timeout=1)
            except Exception as e:
                print(f"Capture error: {str(e)}")

    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.model.conf = self.confidence.get()
                results = self.model(rgb_frame)
                detections = results.pandas().xyxy[0]
                
                if not detections.empty:
                    self.detection_queue.put(detections)
                    if self.voice_var.get():
                        self.voice_queue.put(detections)
                
                display_frame = self.draw_detections(frame, detections)
                self.display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")

    def draw_detections(self, frame, detections):
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = map(int, det[['xmin', 'ymin', 'xmax', 'ymax']])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def update_gui(self):
        if self.running:
            if hasattr(self, 'display_frame'):
                img = Image.fromarray(self.display_frame)
                img.thumbnail((800, 600))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk
            
            try:
                detections = self.detection_queue.get_nowait()
                self.update_log(detections)
            except queue.Empty:
                pass
            
            self.root.after(10, self.update_gui)

    def update_log(self, detections):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        
        current_objects = set()
        for _, det in detections.iterrows():
            obj_name = det['name']
            current_objects.add(obj_name)
            status = "[New] " if obj_name not in self.announced_objects else ""
            entry = f"{status}{obj_name} ({det['confidence']:.2f})\n"
            self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {entry}")
        
        self.log_text.config(state=tk.DISABLED)

    def process_voice(self):
        while self.running:
            try:
                detections = self.voice_queue.get(timeout=1)
                names = set(detections['name'].unique())
                new_objects = names - self.announced_objects
                
                if new_objects and self.voice_enabled:
                    announcement = "New detection: " + ", ".join(new_objects)
                    subprocess.call(['say', '-v', 'Samantha', announcement])
                    self.announced_objects.update(new_objects)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice error: {str(e)}")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Stop detection?"):
            self.stop_detection()
            self.cap.release()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    detector = MacObjectDetector(root)
    root.protocol("WM_DELETE_WINDOW", detector.on_closing)
    root.mainloop()