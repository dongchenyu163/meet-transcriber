"""Capture timing statistics only; no microphone samples are saved."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audio import CONTEXT, _open_stream, describe_error


def check(loopback, output):
    import numpy as np
    recorder = None
    try:
        recorder, rate = _open_stream(None,loopback)
        count = 0
        began = time.perf_counter()
        gaps = []
        for _ in range(32):
            start = time.perf_counter()
            block = recorder.record(rate//4)
            gaps.append(time.perf_counter()-start)
            count += len(block)
        output.put({'source':'system' if loopback else 'mic','rate':rate,'shape':block.shape,
                    'wall_seconds':round(time.perf_counter()-began,3),'audio_seconds':count/rate,
                    'read_min':round(min(gaps),3),'read_max':round(max(gaps),3),
                    'channel_rms':np.sqrt(np.mean(block.astype(float)**2,axis=0)).tolist()})
    except Exception as exc:output.put({'error':describe_error(exc),'loopback':loopback})
    finally:
        if recorder is not None:recorder.__exit__(None,None,None)


if __name__=='__main__':
    output=CONTEXT.Queue()
    procs=[CONTEXT.Process(target=check,args=(loopback,output)) for loopback in (False,True)]
    try:
        for p in procs:p.start()
        for _ in procs:print(output.get(timeout=35),flush=True)
    finally:
        for p in procs:
            p.join(1)
            if p.is_alive():p.terminate();p.join()
        output.close()
        output.join_thread()
