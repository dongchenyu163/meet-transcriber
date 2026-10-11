"""Play the most recent system hardware check via the GUI's Qt audio path."""
from pathlib import Path
import json
import os
import sys
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtMultimedia import QMediaPlayer
from gui import MainWindow


if __name__=='__main__':
    app=QApplication([])
    window=MainWindow(load_inventory=False)
    result=json.loads((ROOT/'test-output/hardware/results.json').read_text(encoding='utf-8'))[-1]
    window.hardware_file=result['path']
    errors=[]
    ended=[]
    window.replay_hardware()
    window.player.errorOccurred.connect(lambda error,message:(errors.append(message),app.quit()))
    def status(value):
        if value==QMediaPlayer.EndOfMedia:
            ended.append(True)
            app.quit()
    window.player.mediaStatusChanged.connect(status)
    QTimer.singleShot(15000,app.quit)
    app.exec()
    duration=window.player.duration()
    window.close()
    assert not errors,errors
    assert ended,'Playback did not reach EndOfMedia'
    assert abs(duration/1000-result['seconds'])<.1,duration
    print(f'Qt GUI playback reached EndOfMedia: {duration}ms; PASS')
