"""
Facial Recognition Desktop App - Face Detection
A desktop application for real-time face detection with camera controls.
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime
import numpy as np
from deepface import DeepFace


class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition - Face Detection + Emotion")
        self.root.geometry("1150x850")
        self.root.resizable(True, True)
        
        # Variables
        self.camera_index = 0
        self.cameras = []
        self.cap = None
        self.is_running = False
        self.thread = None
        
        # Load cascade classifiers - use default for better accuracy with glasses
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.prev_time = 0

        # Emotion detection
        self.emotion_every_n_frames = 5
        self.emotion_frame_counter = 0
        self.emotion_confidence_threshold = 0.3
        self.latest_emotions = []
        
        # Setup GUI
        self.setup_styles()
        self.setup_ui()
        self.detect_cameras()
        
    def setup_styles(self):
        """Setup modern UI styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Modern color palette
        self.colors = {
            'bg': '#1e1e2e',           # Dark background
            'surface': '#2a2a3e',      # Card/surface
            'primary': '#6366f1',      # Primary blue
            'primary_hover': '#4f46e5',
            'success': '#10b981',      # Green
            'success_hover': '#059669',
            'danger': '#ef4444',       # Red
            'danger_hover': '#dc2626',
            'text': '#e5e7eb',         # Light text
            'text_secondary': '#9ca3af'
        }
        
        # Configure main window
        self.root.configure(bg=self.colors['bg'])
        
        # Frame styles
        self.style.configure('Main.TFrame', background=self.colors['bg'])
        self.style.configure('Card.TLabelframe', 
                           background=self.colors['surface'],
                           foreground=self.colors['text'],
                           borderwidth=0,
                           relief='flat')
        self.style.configure('Card.TLabelframe.Label',
                           background=self.colors['surface'],
                           foreground=self.colors['text'],
                           font=('Segoe UI', 10, 'bold'))
        
        # Button styles
        self.style.configure('Primary.TButton',
                           background=self.colors['primary'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           font=('Segoe UI', 10),
                           padding=(20, 10))
        self.style.map('Primary.TButton',
                      background=[('active', self.colors['primary_hover']),
                                ('pressed', self.colors['primary_hover'])])
        
        self.style.configure('Success.TButton',
                           background=self.colors['success'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           font=('Segoe UI', 10),
                           padding=(20, 10))
        self.style.map('Success.TButton',
                      background=[('active', self.colors['success_hover']),
                                ('pressed', self.colors['success_hover'])])
        
        self.style.configure('Danger.TButton',
                           background=self.colors['danger'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           font=('Segoe UI', 10),
                           padding=(20, 10))
        self.style.map('Danger.TButton',
                      background=[('active', self.colors['danger_hover']),
                                ('pressed', self.colors['danger_hover'])])
        
        # Combobox style
        self.style.configure('Modern.TCombobox',
                           fieldbackground=self.colors['surface'],
                           background=self.colors['surface'],
                           foreground=self.colors['text'],
                           borderwidth=1,
                           arrowcolor=self.colors['text'])
        
        # Label styles
        self.style.configure('Modern.TLabel',
                           background=self.colors['surface'],
                           foreground=self.colors['text'],
                           font=('Segoe UI', 10))
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15", style='Main.TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="Facial Recognition App", 
                              font=("Segoe UI", 24, "bold"),
                              bg=self.colors['bg'],
                              fg=self.colors['text'])
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        subtitle_label = tk.Label(main_frame, 
                                 text="Face Detection + Emotion Recognition", 
                                 font=("Segoe UI", 12),
                                 bg=self.colors['bg'],
                                 fg=self.colors['text_secondary'])
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Controls Frame
        controls_frame = ttk.LabelFrame(main_frame, text="Camera Controls", 
                                       padding="15", style='Card.TLabelframe')
        controls_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Camera Selection
        ttk.Label(controls_frame, text="Select Camera:", style='Modern.TLabel').grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10), pady=(0, 15))
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(controls_frame, textvariable=self.camera_var, 
                                        state="readonly", width=40, style='Modern.TCombobox')
        self.camera_combo.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        
        # Buttons Frame
        buttons_frame = ttk.Frame(controls_frame, style='Main.TFrame')
        buttons_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.start_btn = ttk.Button(buttons_frame, text="▶ Start Camera", 
                                    command=self.start_camera, style='Success.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(buttons_frame, text="⬛ Stop Camera", 
                                   command=self.stop_camera, state=tk.DISABLED, 
                                   style='Danger.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.capture_btn = ttk.Button(buttons_frame, text="📸 Capture Screenshot", 
                                     command=self.capture_screenshot, state=tk.DISABLED,
                                     style='Primary.TButton')
        self.capture_btn.pack(side=tk.LEFT)
        
        # Video Display Frame
        video_frame = ttk.LabelFrame(main_frame, text="Live Feed", 
                                    padding="10", style='Card.TLabelframe')
        video_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        self.video_label = tk.Label(video_frame, background=self.colors['bg'], 
                                    borderwidth=0, highlightthickness=0)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Info Frame
        info_frame = ttk.LabelFrame(main_frame, text="Detection Stats", 
                                   padding="15", style='Card.TLabelframe')
        info_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=0)
        
        # Status
        ttk.Label(info_frame, text="Status:", style='Modern.TLabel').grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8))
        self.status_label = tk.Label(info_frame, text="Ready", 
                                     fg=self.colors['success'], 
                                     bg=self.colors['surface'],
                                     font=("Segoe UI", 10, "bold"))
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 25))
        
        # Face Count
        ttk.Label(info_frame, text="Faces:", style='Modern.TLabel').grid(
            row=0, column=2, sticky=tk.W, padx=(0, 8))
        self.face_count_label = tk.Label(info_frame, text="0", 
                                         fg=self.colors['primary'], 
                                         bg=self.colors['surface'],
                                         font=("Segoe UI", 10, "bold"))
        self.face_count_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 25))
        
        # FPS
        ttk.Label(info_frame, text="FPS:", style='Modern.TLabel').grid(
            row=0, column=4, sticky=tk.W, padx=(0, 8))
        self.fps_label = tk.Label(info_frame, text="0", 
                                  fg='#a78bfa',  # Purple
                                  bg=self.colors['surface'],
                                  font=("Segoe UI", 10, "bold"))
        self.fps_label.grid(row=0, column=5, sticky=tk.W)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(1, weight=1)
        
    def detect_cameras(self):
        """Detect available cameras"""
        self.cameras = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            self.cameras.append(index)
            cap.release()
            index += 1
        
        if self.cameras:
            camera_names = [f"Camera {i+1}" for i in range(len(self.cameras))]
            self.camera_combo['values'] = camera_names
            self.camera_combo.current(0)
            self.camera_index = self.cameras[0]
            self.update_status(f"Found {len(self.cameras)} camera(s)")
        else:
            self.update_status("No cameras found!")
            messagebox.showerror("Error", "No cameras detected on this system.")
    
    def on_camera_selected(self, event=None):
        """Handle camera selection from dropdown"""
        if self.is_running:
            self.stop_camera()
            self.camera_index = self.cameras[self.camera_combo.current()]
            self.start_camera()
        else:
            self.camera_index = self.cameras[self.camera_combo.current()]
    
    def start_camera(self):
        """Start camera capture"""
        if self.is_running:
            return
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.NORMAL)
            self.camera_combo.config(state=tk.DISABLED)
            
            self.update_status("Camera started - detecting faces")
            
            # Start detection thread
            self.thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
            self.is_running = False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        self.camera_combo.config(state="readonly")
        
        self.update_status("Camera stopped")
        self.video_label.config(image="")
        self.video_label.image = None
        self.face_count_label.config(text="0")
        self.fps_label.config(text="0")
    
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        import time
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces with optimized parameters for speed
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,   # Smaller steps for better accuracy
                    minNeighbors=4,     # Lower threshold for glasses
                    minSize=(40, 40),   # Detect smaller faces
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                self.emotion_frame_counter += 1
                run_emotion = (self.emotion_frame_counter % self.emotion_every_n_frames == 0)
                if run_emotion:
                    self.latest_emotions = []

                # Draw face rectangles and eyes
                for i, (x, y, w, h) in enumerate(faces):
                    # Draw face rectangle
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw face label with index
                    label = f"Face {i+1}"
                    cv2.putText(frame, label, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    emotion_label = None
                    if run_emotion:
                        emotion_label = self.detect_emotion_for_face(frame, x, y, w, h)
                        self.latest_emotions.append(emotion_label)
                    elif i < len(self.latest_emotions):
                        emotion_label = self.latest_emotions[i]

                    if emotion_label:
                        text_y = y + h + 22
                        if text_y > frame.shape[0] - 10:
                            text_y = max(y - 10, 10)
                        cv2.putText(frame, emotion_label, (x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Detect eyes within face region (works with glasses)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(
                        roi_gray, 
                        scaleFactor=1.2, 
                        minNeighbors=8,  # Higher to reduce false positives
                        minSize=(20, 20)
                    )
                    
                    for (ex, ey, ew, eh) in eyes[:2]:  # Usually 2 eyes
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                
                # Update face count
                self.face_count_label.config(text=str(len(faces)))
                
                # Calculate FPS
                current_time = time.time()
                if self.prev_time > 0:
                    self.fps = int(1 / (current_time - self.prev_time))
                self.prev_time = current_time
                self.fps_label.config(text=str(self.fps))
                
                # Update status
                if len(faces) > 0:
                    self.update_status(f"Detecting {len(faces)} face(s) with emotion")
                else:
                    self.update_status("No faces detected")
                
                # Convert BGR to RGB and resize for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display (fit in window)
                display_height = 500
                ratio = display_height / rgb_frame.shape[0]
                display_width = int(rgb_frame.shape[1] * ratio)
                rgb_frame = cv2.resize(rgb_frame, (display_width, display_height))
                
                # Convert to PhotoImage
                image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update label
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)  # Minimal delay for better FPS
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                break

    def detect_emotion_for_face(self, frame, x, y, w, h):
        """Detect emotion for a face crop and return a label string."""
        face_bgr = frame[y:y + h, x:x + w]
        if face_bgr.size == 0:
            return None

        try:
            result = DeepFace.analyze(face_bgr, actions=['emotion'], enforce_detection=False)
            if result and len(result) > 0:
                emotions = result[0]['emotion']
                dominant_emotion = result[0]['dominant_emotion']
                confidence = emotions.get(dominant_emotion, 0)
                if confidence < self.emotion_confidence_threshold:
                    return f"Uncertain ({confidence:.2f})"
                return f"{dominant_emotion} ({confidence:.2f})"
        except Exception as e:
            return None
        return None
    
    def capture_screenshot(self):
        """Capture and save current frame"""
        if not self.cap or not self.is_running:
            messagebox.showwarning("Warning", "Camera is not running")
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Create screenshots directory if it doesn't exist
            screenshot_dir = "screenshots"
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshot_dir, f"face_detection_{timestamp}.png")
            
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Success", f"Screenshot saved:\n{filename}")
            self.update_status(f"Screenshot saved: {filename}")
        else:
            messagebox.showerror("Error", "Failed to capture frame")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_camera()
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
