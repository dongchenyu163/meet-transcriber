@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call setup.bat
    if errorlevel 1 exit /b 1
)
if exist requirements-gpu.lock.txt (
    ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.lock.txt
) else (
    ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
)
if errorlevel 1 (
    pause
    exit /b 1
)
echo GPU runtime installed in the project virtual environment.
echo Restart Meet Notes before selecting NVIDIA GPU.
