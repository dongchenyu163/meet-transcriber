from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from audio import AudioStore
from session import MeetingSession, SessionConfig, retranscribe
from storage import TranscriptWriter
from transcribe import _transcription_worker, TranscriptLine
import queue
import numpy as np


class FakeProcess:
    pid = None
    exitcode = None
    alive = False

    def is_alive(self): return self.alive


class FakeCapture:
    instances = []
    def __init__(self, store, device, loopback):
        self.store = store
        self.ready = threading.Event()
        self.error, self.forced = None, False
        self.started = self.stopped = self.joined = False
        self.loopback = loopback
        self.instances.append(self)

    def start(self):
        self.started = True
        self.store.set_started_at(1000)
        self.store.push(np.ones(16000, dtype=np.float32) * .1)
        self.ready.set()

    def poll_error(self): return self.error
    def stop(self): self.stopped = True
    def join(self): self.joined = True


class FakeASR:
    instances = []
    def __init__(self, stores, options):
        self.stores = stores
        self.ready = threading.Event()
        self.capture_done = threading.Event()
        self.cancel = threading.Event()
        self.process = FakeProcess()
        self.emitted = False
        self.closed = self.terminated = False
        self.instances.append(self)

    def start(self):
        self.process.pid = 100
        self.process.alive = True
        self.ready.set()

    def drain(self):
        if self.capture_done.is_set() and not self.emitted:
            self.emitted = True
            assert all(c.joined for c in FakeCapture.instances)
            self.process.alive = False
            self.process.exitcode = 0
            return [('line', TranscriptLine(1000, 1001, 'final tail', source)) for source in self.stores]
        return []

    def terminate(self):
        self.terminated = True
        self.process.alive = False
        self.process.exitcode = -1

    def close(self): self.closed = True


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='meet-session-tests-')
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        FakeCapture.instances = []
        FakeASR.instances = []

    def tearDown(self): self.temp.cleanup()

    def run_session(self, emit, asr=FakeASR, capture=FakeCapture):
        session = MeetingSession(SessionConfig(out_dir=str(self.root)))
        session.emit = lambda k, v: emit(session, k, v)
        with patch('session.AudioCapture', capture), patch('session.TranscriptionProcess', asr):
            result = session.run()
        data = json.loads((Path(result) / 'transcript.json').read_text(encoding='utf-8'))
        return session, data

    def test_capture_stops_before_tail_and_both_sources_persist(self):
        def emit(session, kind, value):
            if kind == 'recording' and value:
                session.stop()
        _, data = self.run_session(emit)
        self.assertEqual(data['status'], 'complete')
        self.assertEqual({line['source'] for line in data['lines']}, {'mic', 'system'})
        self.assertTrue(all(c.stopped and c.joined for c in FakeCapture.instances))
        self.assertFalse(FakeASR.instances[0].terminated)
        self.assertTrue(FakeASR.instances[0].closed)

    def test_save_only_marks_partial_without_second_inference(self):
        def emit(session, kind, value):
            if kind == 'recording' and value:
                session.stop(save_only=True)
        _, data = self.run_session(emit)
        self.assertEqual(data['status'], 'partial')
        self.assertTrue(FakeASR.instances[0].terminated)
        self.assertEqual(data['lines'], [])

    def test_recovery_counts_saved_tracks_instead_of_live_device_checkboxes(self):
        store = AudioStore(self.root / 'system.wav', started_at=1000)
        store.push(np.ones(16000, dtype=np.float32) * .1)
        writer = TranscriptWriter(self.root, 1000)
        writer.finalize(1001, 'partial')
        # One remote participant in a historical system-only recording, even
        # though the current GUI has its microphone checkbox enabled.
        config = SessionConfig(participants=1, record_mic=True)
        with patch('session.TranscriptionProcess', FakeASR):
            result = retranscribe(self.root, config, lambda *_: None, threading.Event())
        data = json.loads((Path(result) / 'transcript.json').read_text(encoding='utf-8'))
        self.assertEqual(data['status'], 'complete')
        self.assertEqual(data['system_speaker_count'], 1)
        self.assertTrue(config.record_mic)

    def test_device_failure_is_immediate_and_final_status_is_failed(self):
        class BrokenCapture(FakeCapture):
            def poll_error(self): return 'device disconnected'
        messages = []
        _, data = self.run_session(lambda s, k, v: messages.append((k, v)), capture=BrokenCapture)
        self.assertEqual(data['status'], 'failed')
        self.assertTrue(any(k == 'error' and 'disconnected' in v for k, v in messages))
        self.assertTrue(all(c.stopped for c in FakeCapture.instances))

    def test_model_failure_never_starts_recording(self):
        class BrokenASR(FakeASR):
            def start(self):
                self.process.pid = 100
                self.process.alive = False
            def drain(self): return [('error', 'model unavailable')]
        _, data = self.run_session(lambda *args: None, asr=BrokenASR)
        self.assertEqual(data['status'], 'failed')
        self.assertFalse(any(c.started for c in FakeCapture.instances))

    def test_worker_load_failure_reaches_coordinator_queue(self):
        output = queue.Queue()
        ready, done, cancel = threading.Event(), threading.Event(), threading.Event()
        with patch('transcribe.Transcriber.load', side_effect=RuntimeError('load failed')):
            _transcription_worker({}, {}, output, ready, done, cancel)
        kind, value = output.get_nowait()
        self.assertEqual(kind, 'error')
        self.assertIn('load failed', value)
        self.assertFalse(ready.is_set())

    def test_speaker_worker_runs_during_recording_and_finishes_with_asr(self):
        class FakeSpeaker(FakeASR):
            def __init__(self, store, num_speakers=0):
                super().__init__({'system':store}, {})
                self.sent = False
            def drain(self):
                result = []
                if self.stores['system'].snapshot()[0] and not self.sent:
                    self.sent = True
                    result.append(('speakers', {'start':1000,'end':1001,'turns':[
                        {'start':1000,'end':1001,'speaker':'SPEAKER_01','speaker_name':None}]}))
                if self.capture_done.is_set() and self.process.alive:
                    self.process.alive = False
                    self.process.exitcode = 0
                    result.append(('speaker_done',None))
                return result
        saw_live_speakers = []
        session = MeetingSession(SessionConfig(out_dir=str(self.root),diarize=True))
        def emit(kind,value):
            if kind == 'transcript_refresh':
                saw_live_speakers.append(not session.stop_event.is_set())
            if kind == 'recording' and value:
                session.stop()
        session.emit = emit
        with patch('session.AudioCapture',FakeCapture), patch('session.TranscriptionProcess',FakeASR), patch('session.SpeakerProcess',FakeSpeaker):
            path = session.run()
        data = json.loads((Path(path)/'transcript.json').read_text(encoding='utf-8'))
        self.assertEqual(saw_live_speakers,[True])
        self.assertEqual(data['status'],'complete')
        self.assertEqual(data['speaker_status'],'complete')
        self.assertEqual(next(line for line in data['lines'] if line['source']=='system')['speaker'],'SPEAKER_01')

    def test_speaker_failure_does_not_stop_recording_or_transcription(self):
        class BrokenSpeaker(FakeASR):
            def __init__(self,store,num_speakers=0): super().__init__({'system':store},{})
            def start(self):
                self.process.pid = 100
                self.process.alive = False
                self.process.exitcode = 1
            def drain(self):
                if not self.emitted:
                    self.emitted = True
                    return [('speaker_error','test speaker model failure')]
                return []
        session = MeetingSession(SessionConfig(out_dir=str(self.root),diarize=True))
        session.emit = lambda kind,value: session.stop() if kind=='recording' and value else None
        with patch('session.AudioCapture',FakeCapture), patch('session.TranscriptionProcess',FakeASR), patch('session.SpeakerProcess',BrokenSpeaker):
            path = session.run()
        data = json.loads((Path(path)/'transcript.json').read_text(encoding='utf-8'))
        self.assertEqual(data['status'],'complete')
        self.assertEqual(data['speaker_status'],'failed')
        self.assertEqual(len(data['lines']),2)


if __name__ == '__main__':
    unittest.main()
