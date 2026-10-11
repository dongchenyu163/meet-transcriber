"""Exercise both real models concurrently on a public fixture, never a microphone."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio import AudioStore
from speakers import SpeakerProcess
from transcribe import TranscriptionProcess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['whisper','sensevoice'],default='whisper')
    args=parser.parse_args()
    from faster_whisper.audio import decode_audio
    root = Path(__file__).resolve().parents[1] / 'test-output'
    samples = decode_audio(str(root / 'four-speakers-zh.wav'), sampling_rate=16000)
    store = AudioStore(root / 'parallel-fixture.wav', started_at=1000)
    asr = TranscriptionProcess({'system': store}, dict(model_size='large-v3', device='cuda', compute_type='float16',
        language='zh',backend=args.backend,sensevoice_variant='small-fp32'))
    speaker = SpeakerProcess(store,4)
    peak = 0
    lines, turns, errors = [], [], []
    started = time.perf_counter()
    asr.start()
    speaker.start()
    try:
        deadline = started + 180
        while not (asr.ready.is_set() and speaker.ready.is_set()):
            for process in (asr, speaker):
                for kind, value in process.drain():
                    if 'error' in kind:
                        errors.append(value)
                if not process.process.is_alive():
                    raise RuntimeError(str(errors) or 'worker exited during load')
            if time.perf_counter() > deadline:
                raise TimeoutError('model load timeout')
            time.sleep(.1)
        loaded = time.perf_counter()
        store.push(samples)
        asr.capture_done.set()
        speaker.capture_done.set()
        speaker_finished = None
        while asr.process.is_alive() or speaker.process.is_alive():
            for kind, value in asr.drain():
                if kind == 'line': lines.append(value.text)
                elif kind == 'error': errors.append(value)
            for kind, value in speaker.drain():
                if kind == 'speakers': turns.extend(value['turns'])
                elif kind == 'speaker_error': errors.append(value)
                elif kind == 'speaker_done': speaker_finished = time.perf_counter()
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode == 0: peak = max(peak, int(result.stdout.strip().splitlines()[0]))
            if time.perf_counter() > deadline:
                raise TimeoutError('inference timeout')
            time.sleep(.1)
        for kind, value in asr.drain():
            if kind == 'line': lines.append(value.text)
            elif kind == 'error': errors.append(value)
        for kind, value in speaker.drain():
            if kind == 'speakers': turns.extend(value['turns'])
            elif kind == 'speaker_error': errors.append(value)
        result = {'audio_seconds': len(samples)/16000, 'load_seconds': loaded-started,
            'parallel_seconds': time.perf_counter()-loaded,
            'speaker_seconds': (speaker_finished-loaded) if speaker_finished else None,
            'gpu_total_peak_mib': peak, 'speakers': sorted({t['speaker'] for t in turns}),
            'turns': turns, 'transcript': lines, 'errors': errors,
            'exitcodes': [asr.process.exitcode, speaker.process.exitcode]}
        (root / f'parallel-results-{args.backend}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('turns','transcript')}, ensure_ascii=False))
        if errors or not lines or len(result['speakers']) < 2:
            raise RuntimeError('parallel smoke test failed')
    finally:
        asr.terminate()
        speaker.terminate()
        asr.close()
        speaker.close()


if __name__ == '__main__':
    main()
