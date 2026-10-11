"""Process-local CUDA DLL discovery; no system PATH or toolkit changes."""
import ctypes
import os
from pathlib import Path
import sys
from importlib.metadata import version

# add_dll_directory registrations must remain alive throughout inference.
_DLL_DIRECTORIES = []
_LOADED_LIBRARIES = []
_PREPARED = False


def cuda_directories():
    paths = []
    nvidia = Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia'
    for package in ('cuda_runtime', 'cuda_nvrtc', 'nvjitlink', 'cufft', 'cublas', 'cudnn'):
        for subdir in ('bin', 'lib'):
            paths.append(nvidia / package / subdir)
    for key, value in os.environ.items():
        if key == 'CUDA_PATH' or key.startswith('CUDA_PATH_V'):
            paths.extend([Path(value) / 'bin', Path(value) / 'bin' / 'x64'])
    result = []
    for path in paths:
        if path.is_dir() and path not in result:
            result.append(path.resolve())
    return result


def prepare_cuda_runtime():
    global _PREPARED
    if sys.platform != 'win32' or _PREPARED:
        return
    paths = cuda_directories()
    # Some libraries use Python's DLL search, others call LoadLibrary directly.
    for path in paths:
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(path)))
    os.environ['PATH'] = os.pathsep.join([*(str(p) for p in paths), os.environ.get('PATH', '')])
    required = ('cublasLt64_12.dll', 'cublas64_12.dll', 'cudnn64_9.dll')
    for name in required:
        candidate = next((path / name for path in paths if (path / name).is_file()), None)
        try:
            _LOADED_LIBRARIES.append(ctypes.WinDLL(str(candidate) if candidate else name))
        except OSError as exc:
            raise RuntimeError(
                f'GPU 运行库不可用：{name}（Windows 错误 {getattr(exc, "winerror", "未知")}）。'
                '当前引擎需要 CUDA 12 的 cuBLAS 和 cuDNN 9；安装 CUDA 13 不会提供这些 DLL。'
                '请运行 setup-gpu.bat 安装项目运行库，或选择 CPU。'
            ) from exc
    _PREPARED = True


def prepare_sensevoice_cuda():
    """Check this engine's CUDA build and provider DLLs before recording starts."""
    if '+cuda12.cudnn9' not in version('sherpa-onnx'):
        raise RuntimeError('SenseVoice GPU 需要 CUDA 版 sherpa-onnx；请运行 setup-sensevoice-gpu.bat 后重启程序。')
    prepare_cuda_runtime()
    if sys.platform != 'win32':
        return
    native = Path(sys.prefix) / 'Lib' / 'site-packages' / 'sherpa_onnx' / 'lib'
    _DLL_DIRECTORIES.append(os.add_dll_directory(str(native)))
    paths = cuda_directories() + [native]
    # ORT must load its provider itself after setting its host callbacks;
    # preloading the CUDA provider directly causes Windows error 1114.
    for name in ('onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll'):
        if not (native/name).is_file():
            raise RuntimeError(f'SenseVoice GPU 缺少 {name}；请运行 setup-sensevoice-gpu.bat。')
    for name in ('cudart64_12.dll', 'nvJitLink_120_0.dll', 'cufft64_11.dll'):
        candidate = next((p/name for p in paths if (p/name).is_file()), None)
        if candidate is None:
            raise RuntimeError(f'SenseVoice GPU 缺少 {name}；请运行 setup-sensevoice-gpu.bat。')
        try:
            _LOADED_LIBRARIES.append(ctypes.WinDLL(str(candidate)))
        except OSError as exc:
            raise RuntimeError(f'SenseVoice GPU 无法加载 {name}：{exc}。请运行 setup-sensevoice-gpu.bat。') from exc
