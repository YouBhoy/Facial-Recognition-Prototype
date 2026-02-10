@echo off
REM Facial Recognition App - One-Click Launch
REM This batch file activates the virtual environment and runs the app

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the app
python facial_recognition_app.py

REM Pause to show any error messages
pause
