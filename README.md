# Facial Recognition App - Face Detection

A desktop application for real-time facial detection using your webcam with multi-camera support and easy-to-use controls.

## Features

✨ **Face Detection**
- Real-time face detection with bounding boxes
- **Works with glasses and various face angles** 🎯
- Facial landmark detection (eyes, nose, mouth, ears)
- Confidence scores for each detected face
- Multiple simultaneous face detection

🎥 **Camera Control**
- Start/Stop camera with a single button click
- Switch between multiple connected cameras via dropdown
- Automatic camera enumeration
- Support for all connected video input devices
- Mirror mode for natural viewing

📸 **Screenshot Capture**
- Capture current video frame with all detections visible
- Screenshots automatically saved with timestamp
- Organized in "screenshots" folder

📊 **Live Statistics**
- Real-time FPS (Frames Per Second) counter
- Live face count display
- Detection status updates
- Responsive UI with status messages

## How to Use

### Installation

1. **Install Python** (version 3.8 or higher)
   - Download from [python.org](https://www.python.org)

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

1. **Start the Application**
   ```bash
   python facial_recognition_app.py
   ```

2. **Grant Camera Permission**
   - Allow the application to access your camera when prompted

3. **Using the App**
   - **Select Camera**: Choose from the dropdown if you have multiple cameras
   - **Start Camera**: Click to begin real-time face detection
   - **Stop Camera**: Click to stop the camera and detection
   - **Capture Screenshot**: Save current frame with detected faces as PNG

4. **Monitor Detection**
   - Watch the live video feed with face detection boxes
   - Check the status panel for:
     - Current detection status
     - Number of faces detected
     - Frames per second (FPS)
     - All screenshots saved in the "screenshots" folder

## Technical Details

### Technology Stack
- **Face Detection**: MediaPipe Face Detection (Google's advanced ML model)
- **Video Processing**: OpenCV 
- **GUI Framework**: Tkinter (built-in with Python)
- **Image Processing**: OpenCV and Pillow
- **Threading**: Python threading for non-blocking UI

### Detection Methods
- **MediaPipe Face Detection**: Advanced neural network-based detection
- **Model Selection 1**: Full range detection model (handles glasses, various angles, lighting)
- **Keypoint Detection**: Detects facial landmarks (eyes, nose, mouth, ears)
- **Real-time Processing**: Runs locally on your machine
- **No Internet Required**: All processing done offline

### Performance
- **Highly Accurate**: Works with glasses, different angles, and various lighting conditions
- **Fast Processing**: 20-30+ FPS on modern systems
- **Robust**: Handles real-world occlusions and variations
- **Efficient**: Optimized for real-time applications
- **Supports 1280x720 video resolution**

## File Structure
```
├── facial_recognition_app.py    # Main desktop application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

The app will create a `screenshots/` folder for saving captured images.

## Troubleshooting

### Camera Not Detected
- Ensure your camera is connected and not in use by another application
- Try unplugging and reconnecting the camera
- Restart the application
- Check Device Manager (Windows) to confirm camera is recognized

### Application Won't Start
- Verify Python 3.8+ is installed: `python --version`
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Try running from Command Prompt/PowerShell with administrator privileges

### Face Not Detected
- Ensure adequate lighting in your environment
- Keep face 0.5 - 1.5 meters away from camera
- Look directly at the camera
- MediaPipe works with glasses, but excessive reflections can affect detection
- Move to different lighting if needed

### ImportError: No module named 'mediapipe'
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`
- Make sure you're using the correct Python environment

## Performance Notes

- Face detection runs entirely on your local machine - no data is sent anywhere
- **MediaPipe provides superior accuracy** compared to traditional cascade classifiers
- Works well with:
  - Glasses and sunglasses ✅
  - Different face angles and poses ✅
  - Various lighting conditions ✅
  - Multiple simultaneous faces ✅
- Typical FPS: 20-30+ depending on system
- Multi-threading ensures responsive UI even during detection
- All processing happens locally for privacy

## Future Enhancements

Phase 2 will include:
- Emotion recognition (happy, sad, angry, surprised, etc.)
- Face recognition and identification
- Performance improvements with deep learning models
- Recording capabilities
- Custom detection sensitivity settings
- GPU acceleration support

## System Requirements

- **OS**: Windows 7+, macOS 10.12+, Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Webcam**: USB camera or built-in camera
- **CPU**: Quad-core processor recommended (dual-core minimum)

## Requirements

See `requirements.txt` for package versions:
- opencv-python
- Pillow
- mediapipe (advanced neural network face detection)

## License

This project is part of the Facial Recognition Prototype series.

## Author

Created for educational and prototyping purposes.
