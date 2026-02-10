# Quick Start Guide

## Setup (First Time Only)

### 1. Install Python
- Download Python 3.8+ from [python.org](https://www.python.org)
- During installation, **check "Add Python to PATH"**

### 2. Install Dependencies

Open Command Prompt or PowerShell in the project folder and run:

```bash
pip install -r requirements.txt
```

## Running the App

1. Open Command Prompt or PowerShell
2. Navigate to the project folder
3. Run:
   ```bash
   python facial_recognition_app.py
   ```

## Using the App

- **Select Camera**: Choose which camera to use from the dropdown (if you have multiple cameras)
- **Start Camera**: Click to begin face detection
- **Stop Camera**: Click to stop the camera feed
- **Capture Screenshot**: Save the current frame with detected faces

Screenshots are automatically saved in a `screenshots` folder in the project directory.

## Troubleshooting

### "Python is not recognized"
- Ensure Python was added to PATH during installation
- Restart your computer
- Try using `python3` instead of `python`

### "No module named 'cv2'"
Run this command:
```bash
pip install --upgrade opencv-python
```

### Application is slow
- This is normal - face detection uses CPU processing
- Try closing other applications
- Lower system background tasks via Task Manager

### Camera won't start
- Ensure no other application is using the camera (check Teams, Zoom, etc.)
- Restart the application
- Restart your computer

## Camera Switching

If you have multiple cameras:
1. Stop the current camera (click "Stop Camera")
2. Select a different camera from the dropdown
3. Click "Start Camera" again

## Taking Screenshots

- Click "Capture Screenshot" while the camera is running
- Images are saved in the `screenshots` folder
- Filename format: `face_detection_YYYYMMDD_HHMMSS.png`

## Next Steps

Phase 2 will add emotion recognition capabilities!
