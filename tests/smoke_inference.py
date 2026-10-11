"""Optional offline end-to-end smoke test against an already cached small model.

    .venv\Scripts\python.exe tests\smoke_inference.py
Requires test-output/jfk.flac, the public fixture from:
https://github.com/openai/whisper/blob/main/tests/jfk.flac
"""
from pathlib import Path
from datetime import datetime
import argparse
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
from audio import AudioStore
from storage import TranscriptWriter
from transcribe import TranscriptionProcess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--model', default='small')
    args = parser.parse_args()
    from faster_whisper.audio import decode_audio
    root = ROOT / 'test-output' / ('inference_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True, exist_ok=True)
    audio = decode_audio(str(ROOT / 'test-output' / 'jfk.flac'), sampling_rate=16000)
    store = AudioStore(root / 'system.wav', started_at=1000)
    store.push(audio)
    writer = TranscriptWriter(root, 1000)
    asr = TranscriptionProcess({'system': store}, {'model_size': args.model, 'language': 'en',
        'device': args.device, 'compute_type': 'float16' if args.device == 'cuda' else 'int8'})
    asr.start()
    asr.capture_done.set()
    texts = []
    deadline = time.monotonic() + 90
    try:
        while True:
            for kind, value in asr.drain():
                if kind == 'line':
                    writer.on_line(value)
                    texts.append(value.text)
                if kind == 'error':
                    raise RuntimeError(value)
            if not asr.process.is_alive():
                for kind, value in asr.drain():
                    if kind == 'line':
                        writer.on_line(value)
                        texts.append(value.text)
                    if kind == 'error':
                        raise RuntimeError(value)
                if asr.process.exitcode != 0:
                    raise RuntimeError(f'inference process exited {asr.process.exitcode}')
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Local inference exceeded 90 seconds')
            time.sleep(.1)
        text = ' '.join(texts)
        print('Recognized:', text)
        assert 'country' in text.lower() and 'ask' in text.lower(), text
        writer.finalize(1000 + len(audio) / 16000)
        print(f'Real cached Whisper {args.model}/{args.device} inference and process shutdown: PASS')
    finally:
        if asr.process.is_alive():
            asr.terminate()
        asr.close()


if __name__ == '__main__':
    main()
