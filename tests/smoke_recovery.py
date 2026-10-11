"""Verify full-session recovery and speaker attribution on the public fixture."""
from pathlib import Path
import argparse
import json
import sys
import threading
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio import AudioStore
from session import SessionConfig, retranscribe
from storage import TranscriptWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speakers', type=int, default=0)
    args = parser.parse_args()
    from faster_whisper.audio import decode_audio
    output = Path(__file__).resolve().parents[1] / 'test-output'
    root = output / ('recovery-smoke-' + uuid.uuid4().hex[:8])
    root.mkdir()
    store = AudioStore(root / 'system.wav', started_at=1000)
    store.push(decode_audio(str(output / 'four-speakers-zh.wav'), sampling_rate=16000))
    writer = TranscriptWriter(root, 1000, {'audio_starts': {'system':1000}})
    writer.finalize(1057, 'partial')
    config = SessionConfig(model_size='large-v3',device='cuda',compute_type='float16',language='zh',diarize=True,
                           participants=args.speakers)
    destination = retranscribe(root,config,lambda *_: None,threading.Event())
    data = json.loads((Path(destination)/'transcript.json').read_text(encoding='utf-8'))
    identities = sorted({line['speaker'] for line in data['lines'] if (line.get('speaker') or '').startswith('SPEAKER_')})
    assert data['status'] == 'complete', data['status']
    assert data['speaker_status'] == 'complete', data['speaker_status']
    assert len(identities) == (args.speakers or 4), identities
    assert any(line.get('words') for line in data['lines'])
    print(json.dumps({'destination':destination,'status':data['status'], 'speaker_status':data['speaker_status'],
                      'line_count':len(data['lines']),'speakers':identities},ensure_ascii=False))


if __name__ == '__main__': main()
