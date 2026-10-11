import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

import httpx
import numpy as np

from audio import AudioStore, CONTEXT, _capture_worker, _open_stream
from screenshot import ScreenshotResult, Screenshotter, _capture_worker as shot_worker
from settings import DeepSeekSettings
from storage import TranscriptWriter, recover_exports, read_events
from summary import DeepSeekSummarizer, SummaryCancelled, split_text
from transcribe import StreamingTranscriber, TranscriptLine


def synthetic_producer(store):
    store.push(np.ones(1600, dtype=np.float32) * .3)


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(audio.copy())
        # One word per second, with stable word boundaries across overlapping windows.
        duration = audio.size / 16000
        return [(i, min(i + 1, duration), f' word{i}') for i in range(int(np.ceil(duration)))]


class TempCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='meet-tests-')
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))

    def tearDown(self):
        self.temp.cleanup()

    def store(self, seconds, epoch=1000):
        store = AudioStore(self.root / 'system.wav', started_at=epoch)
        store.push(np.full(round(seconds * 16000), .1, dtype=np.float32))
        return store


class AudioTests(TempCase):
    def test_long_backlog_is_read_in_order_without_loss(self):
        store = self.store(90)
        model, lines = FakeModel(), []
        processor = StreamingTranscriber(store, model, lines.append)
        processor.flush()
        self.assertEqual(processor.cursor, 90 * 16000)
        self.assertEqual(len(' '.join(line.text for line in lines).split()), 90)
        self.assertEqual(sum(line.end - line.start for line in lines), 90)
        self.assertEqual(lines[0].start, 1000)
        self.assertEqual(lines[-1].end, 1090)
        self.assertTrue(all(line.end <= next_line.start for line, next_line in zip(lines, lines[1:])))

    def test_flush_transcribes_short_tail(self):
        store = self.store(2.5)
        model, lines = FakeModel(), []
        processor = StreamingTranscriber(store, model, lines.append)
        self.assertFalse(processor.step())
        processor.flush()
        self.assertEqual(processor.cursor, 40000)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(lines[-1].end, 1002.5)

    def test_live_then_flush_preserves_boundary_once(self):
        store = self.store(5)
        model, lines = FakeModel(), []
        processor = StreamingTranscriber(store, model, lines.append)
        self.assertTrue(processor.step())
        store.push(np.full(2 * 16000, .1, dtype=np.float32))
        processor.flush()
        self.assertEqual(processor.cursor, 7 * 16000)
        self.assertEqual(len(' '.join(x.text for x in lines).split()), 7)

    def test_silent_tail_is_committed(self):
        store = AudioStore(self.root / 'silent.wav')
        store.push(np.zeros(1000, dtype=np.float32))
        model = FakeModel()
        processor = StreamingTranscriber(store, model, lambda _: None)
        processor.flush()
        self.assertEqual(processor.cursor, 1000)
        self.assertEqual(model.calls, [])

    def test_wave_is_playable_before_completion_and_after_repair(self):
        store = self.store(3)
        with wave.open(str(store.path)) as f:
            self.assertEqual(f.getnframes(), 48000)
        with store.path.open('r+b') as f:
            f.seek(40)
            f.write(bytes(4))
        recovered = AudioStore(store.path, existing=True)
        self.assertEqual(recovered.snapshot()[0], 48000)

    def test_atomic_snapshot_under_concurrent_push(self):
        store = self.store(1)
        failures = []
        def produce():
            try:
                for _ in range(20):
                    store.push(np.ones(100, dtype=np.float32) * .2)
            except Exception as exc:
                failures.append(exc)
        thread = threading.Thread(target=produce)
        thread.start()
        while thread.is_alive():
            samples, epoch = store.read(0, 16000)
            self.assertEqual(epoch, 1000)
            self.assertEqual(samples.size, 16000)
        thread.join()
        self.assertFalse(failures)

    def test_spawned_producer_shares_store_counter(self):
        store = AudioStore(self.root / 'shared.wav')
        proc = CONTEXT.Process(target=synthetic_producer, args=(store,))
        proc.start()
        proc.join(15)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            self.fail('spawned producer failed to exit')
        self.assertEqual(proc.exitcode, 0)
        self.assertEqual(store.snapshot()[0], 1600)

    def test_device_failure_is_reported(self):
        output = queue.Queue()
        ready, stop = threading.Event(), threading.Event()
        with patch('audio._open_stream', side_effect=RuntimeError('device disconnected')):
            _capture_worker(self.store(0), None, True, stop, ready, output)
        self.assertIn('device disconnected', output.get_nowait())
        self.assertFalse(ready.is_set())

    def test_microphone_and_system_are_selected_independently(self):
        from types import SimpleNamespace
        calls = []
        class Recorder:
            def __enter__(self): return self
        class Device:
            def __init__(self, id, loopback):
                self.id, self.name, self.isloopback, self.channels = id, id, loopback, 2
            def recorder(self, **kwargs):
                calls.append(self.id)
                return Recorder()
        mic, speaker = Device('mic', False), Device('speaker', True)
        module = SimpleNamespace(all_microphones=lambda **kw: [mic, speaker],
            default_speaker=lambda: speaker, default_microphone=lambda: mic)
        with patch.dict('sys.modules', {'soundcard': module}), \
             patch('audio._open_microphone', side_effect=lambda m: calls.append(m.id)):
            _open_stream(None, True)
            _open_stream(None, False)
        self.assertEqual(calls, ['speaker', 'mic'])


class StorageTests(TempCase):
    def test_existing_journal_cannot_be_accidentally_reused(self):
        TranscriptWriter(self.root, 1000)
        with self.assertRaises(FileExistsError):
            TranscriptWriter(self.root, 2000)

    def test_json_and_unclaimed_screenshot_are_saved_immediately(self):
        writer = TranscriptWriter(self.root, 1000)
        writer.add_screenshot_event(ScreenshotResult('shot.png', 'window', 'Meet', 1002))
        data = json.loads((self.root / 'transcript.json').read_text(encoding='utf-8'))
        self.assertEqual(data['screenshots'][0]['path'], 'shot.png')
        self.assertEqual(data['lines'], [])
        self.assertIn('shot.png', (self.root / 'transcript.txt').read_text(encoding='utf-8'))

    def test_late_screenshot_links_to_existing_line(self):
        writer = TranscriptWriter(self.root, 1000)
        writer.on_line(TranscriptLine(1001, 1004, 'hello'))
        writer.add_screenshot_event(ScreenshotResult('late.png', 'window', '', 1003))
        data = recover_exports(self.root)
        self.assertEqual(data['lines'][0]['screenshots'], ['late.png'])

    def test_interrupted_journal_recovers_partial_final_record(self):
        writer = TranscriptWriter(self.root, 1000)
        writer.on_line(TranscriptLine(1001, 1002, 'retained'))
        with writer.journal.open('ab') as f:
            f.write(b'{"type":')
        data = recover_exports(self.root)
        self.assertEqual(data['lines'][0]['text'], 'retained')
        self.assertEqual(data['status'], 'interrupted')

    def test_exports_order_both_sources_by_time(self):
        writer = TranscriptWriter(self.root, 1000)
        writer.on_line(TranscriptLine(1020, 1021, 'later', 'mic'))
        writer.on_line(TranscriptLine(1010, 1011, 'earlier', 'system'))
        writer.finalize(1030)
        data = recover_exports(self.root)
        self.assertEqual([x['text'] for x in data['lines']], ['earlier', 'later'])
        self.assertEqual(data['status'], 'complete')


class ScreenshotTests(TempCase):
    def test_window_missing_never_falls_back_to_desktop(self):
        with self.assertRaisesRegex(RuntimeError, '选择会议窗口'):
            Screenshotter(self.root).capture()

    def test_same_timestamp_still_produces_unique_files(self):
        from PIL import Image
        output = queue.Queue()
        with patch('screenshot._capture_window', return_value=(Image.new('RGB', (4, 4), 'red'), 'Meet')), \
             patch('screenshot.time.time', return_value=1000):
            shot_worker(self.root, 1, False, None, output)
            shot_worker(self.root, 1, False, None, output)
        one, two = output.get_nowait(), output.get_nowait()
        self.assertEqual(one[0], 'ok')
        self.assertEqual(two[0], 'ok')
        self.assertNotEqual(one[1].path, two[1].path)
        self.assertEqual(len(list(self.root.glob('*.png'))), 2)


class SummaryTests(TempCase):
    def setUp(self):
        super().setUp()
        self.settings = DeepSeekSettings('test-secret')
        writer = TranscriptWriter(self.root, 1000)
        writer.on_line(TranscriptLine(1001, 1002, 'Alice will deliver Friday.'))
        writer.finalize(1003)

    def client(self, handler):
        return DeepSeekSummarizer(self.settings, transport=httpx.MockTransport(handler))

    def test_api_receives_only_transcript_and_saves_minutes(self):
        def handle(request):
            self.assertEqual(str(request.url), 'https://api.deepseek.com/chat/completions')
            self.assertEqual(request.headers['authorization'], 'Bearer test-secret')
            body = json.loads(request.content)
            self.assertIn('Alice', body['messages'][1]['content'])
            self.assertNotIn('audio', body)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '# Minutes\nDone'}}]})
        result = self.client(handle).generate(self.root)
        self.assertIn('Minutes', result)
        self.assertTrue((self.root / 'minutes.md').exists())
        self.assertNotIn('test-secret', (self.root / 'events.jsonl').read_text(encoding='utf-8'))

    def test_auth_error_does_not_leak_key_or_destroy_transcript(self):
        def handle(request):
            return httpx.Response(401, text='echo secret test-secret')
        with self.assertRaises(RuntimeError) as context:
            self.client(handle).generate(self.root)
        self.assertNotIn('test-secret', str(context.exception))
        self.assertTrue((self.root / 'transcript.json').exists())
        self.assertFalse((self.root / 'minutes.md').exists())

    def test_truncated_output_is_not_published(self):
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': 'partial'}}]}
        with self.assertRaisesRegex(RuntimeError, '未完整结束'):
            self.client(lambda r: httpx.Response(200, json=response)).generate(self.root)
        self.assertFalse((self.root / 'minutes.md').exists())

    def test_long_text_map_reduce_and_cache(self):
        data = json.loads((self.root / 'transcript.json').read_text(encoding='utf-8'))
        data['lines'][0]['text'] = '会议讨论事项。' * 6000
        (self.root / 'transcript.json').write_text(json.dumps(data), encoding='utf-8')
        requests = []
        def handle(request):
            requests.append(request)
            body = json.loads(request.content)
            self.assertLess(len(body['messages'][1]['content']), 18500)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '简短纪要 [00:16:41]'}}]})
        client = self.client(handle)
        client.generate(self.root)
        count = len(requests)
        self.assertGreaterEqual(count, 3)
        client.generate(self.root)
        self.assertEqual(count, len(requests))

    def test_cancel_prevents_network_request(self):
        cancel = threading.Event()
        cancel.set()
        client = DeepSeekSummarizer(self.settings, cancel=cancel)
        with self.assertRaises(SummaryCancelled):
            client.generate(self.root)

    def test_cancel_interrupts_in_flight_network_request(self):
        import asyncio
        import time
        cancel = threading.Event()
        async def handle(request):
            cancel.set()
            await asyncio.sleep(10)
            return httpx.Response(200)
        client = DeepSeekSummarizer(self.settings, cancel=cancel, transport=httpx.MockTransport(handle))
        start = time.monotonic()
        with self.assertRaises(SummaryCancelled):
            client.generate(self.root)
        self.assertLess(time.monotonic() - start, 2)

    def test_rate_limit_has_bounded_retry(self):
        requests = []
        def handle(request):
            requests.append(request)
            if len(requests) < 3:
                return httpx.Response(429)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': 'done'}}]})
        client = self.client(handle)
        with patch.object(client.cancel, 'wait', return_value=False):
            self.assertEqual(client.generate(self.root), 'done')
        self.assertEqual(len(requests), 3)

    def test_key_configuration_roundtrip_and_repr(self):
        path = self.root / '.env'
        self.settings.save(path)
        with patch.dict('os.environ', {}, clear=True):
            restored = DeepSeekSettings.load(path)
        self.assertEqual(restored.api_key, 'test-secret')
        self.assertNotIn('test-secret', repr(restored))

    def test_split_preserves_every_character(self):
        text = 'a' * 40000 + '\n中文\n' * 3000
        parts = split_text(text)
        self.assertEqual(''.join(parts), text)
        self.assertTrue(all(len(part) <= 18000 for part in parts))


if __name__ == '__main__':
    unittest.main()
