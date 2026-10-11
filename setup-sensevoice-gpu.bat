@echo off
chcp 65001 >nul
cd /d "%~dp0"
call setup-gpu.bat
if errorlevel 1 exit /b 1
rem Download before changing the working CPU installation.
".venv\Scripts\python.exe" -m pip download --no-deps --dest models\runtime-wheels -r requirements-sensevoice-gpu.txt
if errorlevel 1 (
    pause
    exit /b 1
)
rem CPU core and the CUDA wheel own overlapping files; remove the CPU core first.
".venv\Scripts\python.exe" -m pip uninstall -y sherpa-onnx-core
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --no-index --find-links models\runtime-wheels --no-deps --force-reinstall "sherpa-onnx==1.13.8+cuda12.cudnn9"
if errorlevel 1 (
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 exit /b 1
echo SenseVoice CPU and CUDA runtime is ready in .venv.
echo Restart Meet Notes, select SenseVoice Small FP32 and NVIDIA GPU.
