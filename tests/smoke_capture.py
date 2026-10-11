"""Optional two-second hardware input check. Never stores captured audio."""
from pathlib import Path
import queue
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio import CONTEXT, _open_stream, describe_error


def check_source(loopback, output):
    import numpy as np
    recorder = None
    source = 'system' if loopback else 'mic'
    try:
        recorder, rate = _open_stream(None, loopback)
        frames = recorder.record(numframes=rate * 2)
        assert len(frames) == rate * 2 and np.isfinite(frames).all()
        output.put({'source': source, 'ok': True, 'rate': rate, 'shape': frames.shape,
                    'rms': float(np.sqrt(np.mean(frames.astype(np.float64) ** 2)))})
    except Exception as exc:
        output.put({'source': source, 'ok': False, 'error': describe_error(exc)})
    finally:
        if recorder is not None:
            recorder.__exit__(None, None, None)


def main():
    output = CONTEXT.Queue()
    processes = [CONTEXT.Process(target=check_source, args=(loopback, output)) for loopback in (True, False)]
    results = []
    try:
        for process in processes:
            process.start()
        for _ in processes:
            results.append(output.get(timeout=20))
        for result in results:
            print(result)
        assert all(result['ok'] for result in results), 'Hardware capture failed'
        print('Both hardware audio sources: PASS; no recording written.')
    finally:
        for process in processes:
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join()
        output.close()
        output.join_thread()


if __name__ == '__main__':
    main()
