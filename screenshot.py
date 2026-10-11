"""Explicit window capture. Never silently replace a window with the desktop."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import queue
import time
import uuid
from audio import CONTEXT


@dataclass
class ScreenshotResult:
    path: str
    mode: str
    window_title: str
    timestamp: float


def list_windows():
    import win32gui
    result = []
    def visit(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if title and win32gui.IsWindowVisible(hwnd) and not win32gui.IsIconic(hwnd):
            result.append((hwnd, title))
    win32gui.EnumWindows(visit, None)
    return result


def _capture_window(hwnd, expected_pid=None):
    import ctypes
    from ctypes import wintypes
    import win32gui
    import win32ui
    import win32process
    from PIL import Image, ImageStat
    if not hwnd or not win32gui.IsWindow(hwnd) or win32gui.IsIconic(hwnd):
        raise RuntimeError('截图窗口已关闭或最小化，请重新选择窗口。')
    if expected_pid and win32process.GetWindowThreadProcessId(hwnd)[1] != expected_pid:
        raise RuntimeError('截图窗口已变化，请重新选择窗口。')
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top
    if width < 50 or height < 50:
        raise RuntimeError('截图窗口尺寸无效。')
    dc_handle = win32gui.GetWindowDC(hwnd)
    src = win32ui.CreateDCFromHandle(dc_handle)
    target = src.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    previous = None
    try:
        bitmap.CreateCompatibleBitmap(src, width, height)
        previous = target.SelectObject(bitmap)
        print_window = ctypes.windll.user32.PrintWindow
        print_window.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        print_window.restype = wintypes.BOOL
        if not print_window(hwnd, target.GetSafeHdc(), 2):
            raise RuntimeError('窗口不支持截图，请检查窗口是否正常显示。')
        img = Image.frombuffer('RGB', (width, height), bitmap.GetBitmapBits(True),
                               'raw', 'BGRX', 0, 1).copy()
        if max(ImageStat.Stat(img).mean) < 1:
            raise RuntimeError('窗口返回黑屏，请恢复窗口或调整浏览器硬件加速设置后重试。')
        return img, win32gui.GetWindowText(hwnd)
    finally:
        if previous is not None:
            target.SelectObject(previous)
        win32gui.DeleteObject(bitmap.GetHandle())
        target.DeleteDC()
        src.DeleteDC()
        win32gui.ReleaseDC(hwnd, dc_handle)


def _capture_worker(out_dir, hwnd, full_screen, pid, output):
    try:
        from PIL import Image
        timestamp = time.time()
        if full_screen:
            import mss
            with mss.mss() as sct:
                frame = sct.grab(sct.monitors[0])
                img = Image.frombytes('RGB', frame.size, frame.rgb)
            mode, title = 'full', ''
        else:
            img, title = _capture_window(hwnd, pid)
            mode = 'window'
        path = Path(out_dir) / f'{datetime.fromtimestamp(timestamp):%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        output.put(('ok', ScreenshotResult(str(path.resolve()), mode, title, timestamp)))
    except Exception as exc:
        output.put(('error', str(exc)))


class Screenshotter:
    def __init__(self, out_dir, hwnd=None, full_screen=False):
        self.out_dir = Path(out_dir)
        self.hwnd, self.full_screen = hwnd, full_screen
        self.pid = None
        if hwnd:
            import win32process
            self.pid = win32process.GetWindowThreadProcessId(hwnd)[1]

    def capture(self):
        if not self.full_screen and not self.hwnd:
            raise RuntimeError('请先选择会议窗口；截图不会自动切换为整屏。')
        output = CONTEXT.Queue()
        process = CONTEXT.Process(target=_capture_worker,
            args=(str(self.out_dir), self.hwnd, self.full_screen, self.pid, output), daemon=True)
        process.start()
        try:
            try:
                kind, result = output.get(timeout=8)
            except queue.Empty:
                raise RuntimeError('窗口截图超时，请重新选择窗口。') from None
            if kind != 'ok':
                raise RuntimeError(result)
            return result
        finally:
            process.join(.5)
            if process.is_alive():
                process.terminate()
                process.join()
            output.close()
            output.join_thread()
