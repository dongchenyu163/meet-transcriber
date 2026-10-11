@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Running setup...
    call setup.bat
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
