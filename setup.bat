@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Please install Python 3.11, then run setup.bat again.
        pause
        exit /b 1
    )
)
if exist requirements.lock.txt (
    ".venv\Scripts\python.exe" -m pip install -r requirements.lock.txt
) else (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 (
    pause
    exit /b 1
)
if not exist .env copy .env.example .env >nul
echo Setup complete. Run run.bat to open the application.
