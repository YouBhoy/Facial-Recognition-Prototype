# Facial Recognition App - Face Detection

A desktop application for real-time facial detection using your webcam with multi-camera support and easy-to-use controls.

## Features

✨ **Face Detection**
- Real-time face detection with bounding boxes
- Eye detection within detected face regions
- Visual indicators for each detected face
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
- **Face Detection**: OpenCV (Haar Cascade Classifier)
- **Eye Detection**: OpenCV (Haar Cascade Eye Detection)
- **GUI Framework**: Tkinter (built-in with Python)
- **Image Processing**: OpenCV and Pillow
- **Threading**: Python threading for non-blocking UI

### Detection Methods
- **Haar Cascade Frontal Face**: Fast and reliable frontal face detection
- **Haar Cascade Eyes**: Detects eyes within detected face regions
- **Real-time Processing**: Runs locally on your machine
- **No Internet Required**: All processing done offline

### Performance
- Efficient cascade classifiers for fast detection
- Threading prevents UI freezing during detection
- 30 FPS target (actual FPS varies by system)
- Supports 1280x720 video resolution

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

### Poor Detection Performance
- Ensure adequate lighting in your environment
- Keep faces 0.5 - 2 meters away from camera
- This is normal for Haar Cascade - it uses CPU-based detection
- Close other applications to free up system resources

### Slow Performance / High CPU Usage
- This is expected with Haar Cascade detection
- Lower camera resolution for faster processing
- Consider upgrading system CPU for better FPS
- GPU acceleration is not available with Haar Cascade

### ImportError: No module named 'cv2'
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`
- Make sure you're using the correct Python environment

## Performance Notes

- Face detection runs entirely on your local machine - no data is sent anywhere
- Uses CPU-based Haar Cascade detection (highly portable, works on any system)
- FPS varies based on:
  - CPU capabilities
  - Number of faces in frame
  - Video resolution
  - System resource availability
- Typical FPS: 15-30 depending on system
- Multi-threading ensures responsive UI even during heavy detection

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
- **CPU**: Quad-core processor recommended

## Requirements

See `requirements.txt` for package versions:
- opencv-python
- Pillow

## License

This project is part of the Facial Recognition Prototype series.

## Author

Created for educational and prototyping purposes.
