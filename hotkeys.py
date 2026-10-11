"""Windows RegisterHotKey with explicit registration/unregistration."""
import ctypes
from ctypes import wintypes
import sys
from PySide6.QtCore import QAbstractNativeEventFilter


class GlobalHotkeys(QAbstractNativeEventFilter):
    def __init__(self, app, on_shot, on_stop):
        super().__init__()
        self.app = app
        self.callbacks = {0x4D01: on_shot, 0x4D02: on_stop}
        self.registered = []

    def install(self):
        if sys.platform != 'win32':
            return ['全局热键仅支持 Windows，请使用界面按钮。']
        self.app.installNativeEventFilter(self)
        warnings = []
        user32 = ctypes.windll.user32
        for ident, key, label in [(0x4D01, 0x78, 'F9'), (0x4D02, 0x77, 'F8')]:
            if user32.RegisterHotKey(None, ident, 0x4000, key):
                self.registered.append(ident)
            else:
                warnings.append(f'{label} 已被占用或无法注册，请使用界面按钮。')
        return warnings

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) in (b'windows_generic_MSG', b'windows_dispatcher_MSG'):
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0312 and msg.wParam in self.registered:
                self.callbacks[msg.wParam]()
                return True, 0
        return False, 0

    def close(self):
        if sys.platform == 'win32':
            for ident in self.registered:
                ctypes.windll.user32.UnregisterHotKey(None, ident)
        self.registered.clear()
        self.app.removeNativeEventFilter(self)
