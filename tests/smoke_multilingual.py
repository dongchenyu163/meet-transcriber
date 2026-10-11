"""Offline three-language sample through the actual spawned ASR worker."""
from pathlib import Path
import argparse
import os
import sys
import time
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
from audio import AudioStore,RATE
from local_asr import MODEL_DIR
from transcribe import TranscriptionProcess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['whisper','sensevoice'],default='sensevoice')
    parser.add_argument('--model',default='large-v3')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--variant',choices=['small-int8','small-fp32'],default='small-int8')
    args=parser.parse_args()
    from faster_whisper.audio import decode_audio
    samples=np.concatenate([part for language in ('zh','ja','en') for part in
        (decode_audio(str(MODEL_DIR/'test_wavs'/f'{language}.wav'),sampling_rate=RATE),np.zeros(RATE,np.float32))])
    with tempfile.TemporaryDirectory(prefix='meet-multilingual-') as folder:
        store=AudioStore(Path(folder)/'system.wav',started_at=1000)
        store.push(samples)
        asr=TranscriptionProcess({'system':store},dict(backend=args.backend,model_size=args.model,
            device=args.device,compute_type='float16' if args.device=='cuda' else 'int8',multilingual=True,
            sensevoice_variant=args.variant))
        lines=[]
        cursors=[]
        gpu_seen=False
        def drain():
            for kind,value in asr.drain():
                if kind=='error':raise RuntimeError(value)
                if kind=='line':lines.append(value)
                if kind=='cursor':cursors.append(value[1])
        began=time.monotonic()
        asr.start()
        asr.capture_done.set()
        try:
            while asr.process.is_alive():
                drain()
                if args.device=='cuda' and asr.ready.is_set() and not gpu_seen:
                    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                        text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
                    gpu_seen=str(asr.process.pid) in processes.split()
                if time.monotonic()-began>120:raise TimeoutError('ASR smoke test timed out')
                time.sleep(.1)
            drain()
            assert asr.process.exitcode==0,asr.process.exitcode
            text=' '.join(line.text for line in lines)
            print(text,flush=True)
            assert any('\u3040'<=c<='\u30ff' for c in text),text
            assert 'chief' in text.lower(),text
            assert '下午' in text,text
            assert cursors[-1]==len(samples),(cursors[-1],len(samples))
            if args.device=='cuda':assert gpu_seen,'ASR process was not observed using the GPU'
            words=[word for line in lines for word in line.words]
            assert all(1000<=w['start']<w['end']<=1000+len(samples)/RATE for w in words)
            assert all(a['end']<=b['start'] for a,b in zip(words,words[1:]))
            print(f'{args.backend}/{args.variant}/{args.device}: {len(samples)/RATE:.2f}s audio; elapsed {time.monotonic()-began:.2f}s; all three languages and final cursor PASS; GPU observed={gpu_seen}')
        finally:
            if asr.process.is_alive():asr.terminate()
            asr.close()


if __name__=='__main__':main()
