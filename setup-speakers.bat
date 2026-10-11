@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install "sherpa-onnx==1.13.8"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" speaker_models.py
if errorlevel 1 (
    pause
    exit /b 1
)
echo Speaker analysis is ready. Start run.bat.
