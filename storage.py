"""Append-only journal is the source of truth; exports are atomic snapshots."""
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import uuid
from attribution import attribute, speaker_label

SOURCE_LABELS = {'mic': '自己（麦克风）', 'system': '会议声音'}


def atomic_write(path, content):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='\n') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_events(path):
    events = []
    data = Path(path).read_bytes().splitlines()
    for index, raw in enumerate(data):
        try:
            events.append(json.loads(raw))
        except (ValueError, UnicodeDecodeError):
            if index != len(data) - 1:
                raise ValueError('会议事件记录损坏，请保留原文件。')
            # Only the incomplete final record of a crashed writer is ignored.
    return events


def build_snapshot(events):
    meta = next((e['data'].copy() for e in events if e['type'] == 'session'), {})
    lines = [e['data'].copy() for e in events if e['type'] == 'line']
    shots = [e['data'].copy() for e in events if e['type'] == 'screenshot']
    turns, names, revised_turns = [], {}, {}
    for e in events:
        if e['type'] == 'state':
            meta.update(e['data'])
        elif e['type'] == 'audio_start':
            meta.setdefault('audio_starts', {})[e['data']['source']] = e['data']['timestamp']
        elif e['type'] == 'speakers':
            if e['data'].get('upsert'):
                revised_turns.update((turn['id'], turn) for turn in e['data']['turns'])
            else:
                turns.extend(e['data']['turns'])
        elif e['type'] == 'speaker_name':
            names[e['data']['speaker']] = e['data']['name']
    turns.extend(revised_turns.values())
    lines = attribute(lines, turns, names, enabled=bool(meta.get('diarize')))
    lines.sort(key=lambda line: (line['start'], line.get('source', '')))
    shots.sort(key=lambda shot: shot['timestamp'])
    for line in lines:
        line['screenshots'] = [s['path'] for s in shots
                               if line['start'] - .5 <= s['timestamp'] <= line['end'] + .5]
    return {**meta, 'lines': lines, 'screenshots': shots, 'speaker_turns': turns, 'speaker_names': names,
            'errors': [e['data'] for e in events if e['type'] == 'error']}


def export_events(session_dir, events):
    root = Path(session_dir)
    data = build_snapshot(events)
    rows = []
    for line in data['lines']:
        label = speaker_label(line)
        rows.append((line['start'], f'{label}：{line["text"]}'))
    for shot in data['screenshots']:
        rows.append((shot['timestamp'], f'重点截图：{shot["path"]}'))
    rows.sort(key=lambda row: row[0])
    text = ['# 会议转录', f'# 状态：{data.get("status", "interrupted")}', '']
    text.extend(f'[{datetime.fromtimestamp(t):%H:%M:%S}] {content}' for t, content in rows)
    atomic_write(root / 'transcript.json', json.dumps(data, ensure_ascii=False, indent=2))
    atomic_write(root / 'transcript.txt', '\n'.join(text) + '\n')
    return data


class TranscriptWriter:
    def __init__(self, session_dir, started_at, metadata=None):
        self.root = Path(session_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = self.root / 'events.jsonl'
        if self.journal.exists():
            raise FileExistsError('会议目录已有事件记录，请选择新目录或使用恢复功能。')
        self.lock = threading.RLock()
        self.events = []
        self.append('session', {'session_start': started_at, 'status': 'preparing',
                               **(metadata or {})})

    def append(self, kind, data):
        with self.lock:
            event = {'id': uuid.uuid4().hex, 'type': kind, 'data': data}
            with self.journal.open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            self.events.append(event)
            export_events(self.root, self.events)

    def on_line(self, line):
        self.append('line', asdict(line))

    def add_screenshot_event(self, result):
        self.append('screenshot', asdict(result))

    def finalize(self, ended_at, status='complete'):
        self.append('state', {'session_end': ended_at, 'status': status})


def recover_exports(session_dir):
    root = Path(session_dir)
    events = read_events(root / 'events.jsonl')
    data = build_snapshot(events)
    if data.get('status') not in ('complete', 'partial', 'failed'):
        events.append({'type': 'state', 'data': {'status': 'interrupted'}})
    return export_events(root, events)


def rename_speaker(session_dir, speaker, name):
    root = Path(session_dir)
    events = read_events(root / 'events.jsonl')
    if not speaker.startswith('SPEAKER_') or not name.strip() or len(name.strip()) > 80 or '\n' in name:
        raise ValueError('请选择说话人并填写有效姓名。')
    event = {'id': uuid.uuid4().hex, 'type': 'speaker_name',
             'data': {'speaker': speaker, 'name': name.strip()}}
    # History can end in a torn JSON record; rewrite only valid recovered events.
    atomic_write(root / 'events.jsonl', ''.join(json.dumps(e, ensure_ascii=False) + '\n'
                                               for e in events + [event]))
    if (root / 'minutes.md').exists():
        atomic_write(root / '.minutes-stale', 'speaker name changed\n')
    return export_events(root, events + [event])
