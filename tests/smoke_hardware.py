"""User-requested capture test: short mic recording, followed by loopback tones."""
from pathlib import Path
import json
import sys
import threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from soundcheck import HardwareTest


if __name__=='__main__':
    results=[]
    for loopback in (False,True):
        stop=threading.Event()
        timer=None
        def event(kind,value):
            global timer
            if kind=='status' and not loopback:
                timer=threading.Timer(8,stop.set)
                timer.start()
        try:
            result=HardwareTest(loopback,None,stop,event).run()
            print(json.dumps(result,ensure_ascii=False),flush=True)
            results.append(result)
            assert abs(result['seconds']-result['elapsed'])<.8,result
            if loopback:assert result['correlation']>.45,result
        finally:
            if timer:timer.cancel()
    Path('test-output/hardware/results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
