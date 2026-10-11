from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, ANY
import numpy as np

from audio import _open_microphone, MicrophoneRecorder, describe_error
from transcribe import Transcriber, _transcription_worker
import gpu_runtime


class MicrophoneTests(unittest.TestCase):
    def module(self, stream_factory, channels=2):
        return SimpleNamespace(
            query_devices=lambda: [{'name': 'C920', 'hostapi': 0,
                                    'max_input_channels': channels, 'default_samplerate': 48000}],
            query_hostapis=lambda: [{'name': 'Windows WASAPI'}], InputStream=stream_factory)

    def test_pcm_microphone_uses_native_rate_and_portaudio(self):
        stream = Mock()
        factory = Mock(return_value=stream)
        with patch.dict('sys.modules', {'sounddevice': self.module(factory)}):
            recorder, rate = _open_microphone(SimpleNamespace(name='C920'))
        self.assertIsInstance(recorder, MicrophoneRecorder)
        self.assertEqual(rate, 48000)
        factory.assert_called_once_with(device=0, samplerate=48000, channels=2,
                                       dtype='float32', latency='high',callback=ANY)
        self.assertEqual(factory.call_args.kwargs['callback'].__self__,recorder)
        stream.start.assert_called_once()

    def test_mono_microphone_does_not_force_two_channels(self):
        factory = Mock(return_value=Mock())
        with patch.dict('sys.modules', {'sounddevice': self.module(factory, channels=1)}):
            _open_microphone(SimpleNamespace(name='C920'))
        self.assertEqual(factory.call_args.kwargs['channels'], 1)

    def test_failed_start_closes_stream_before_retry(self):
        bad, good = Mock(), Mock()
        bad.start.side_effect = RuntimeError('native rate unavailable')
        factory = Mock(side_effect=[bad, good])
        with patch.dict('sys.modules', {'sounddevice': self.module(factory)}):
            _, rate = _open_microphone(SimpleNamespace(name='C920'))
        self.assertEqual(rate, 16000)
        bad.close.assert_called_once()
        good.start.assert_called_once()

    def test_empty_assertion_is_reported_with_type(self):
        self.assertEqual(describe_error(AssertionError()), 'AssertionError')

    def test_buffer_overflow_is_not_silently_ignored(self):
        recorder=MicrophoneRecorder()
        recorder.callback(np.zeros((10,2),np.float32),10,None,'input overflow')
        with self.assertRaisesRegex(RuntimeError, '丢帧'):
            recorder.record(10)


class GPUPreflightTests(unittest.TestCase):
    def test_sensevoice_cpu_build_is_rejected_for_gpu_before_dll_loading(self):
        with patch('gpu_runtime.version',return_value='1.13.8'), \
             patch('gpu_runtime.prepare_cuda_runtime') as prepare:
            with self.assertRaisesRegex(RuntimeError,'setup-sensevoice-gpu.bat'):
                gpu_runtime.prepare_sensevoice_cuda()
        prepare.assert_not_called()

    def test_sensevoice_preflight_failure_never_announces_recording_ready(self):
        output=queue.Queue()
        ready,done,cancel=(threading.Event() for _ in range(3))
        with patch('local_asr.available',return_value=True), \
             patch('gpu_runtime.prepare_sensevoice_cuda',side_effect=RuntimeError('CUDA unavailable')):
            _transcription_worker({}, {'backend':'sensevoice','device':'cuda','sensevoice_variant':'small-fp32'},
                                  output,ready,done,cancel)
        self.assertFalse(ready.is_set())
        self.assertIn('CUDA unavailable',output.get_nowait()[1])

    def test_cuda_warmup_runs_before_ready(self):
        calls = []
        model = Mock()
        model.feature_extractor.return_value = np.zeros((80, 3000), np.float32)
        model.encode.side_effect = lambda features: calls.append('encode')
        with patch('gpu_runtime.prepare_cuda_runtime', side_effect=lambda: calls.append('dlls')), \
             patch.dict('sys.modules', {'faster_whisper': SimpleNamespace(
                 WhisperModel=lambda *a, **kw: (calls.append('model') or model))}):
            Transcriber(device='cuda', compute_type='float16').load()
        self.assertEqual(calls, ['dlls', 'model', 'encode'])

    def test_missing_cuda_component_never_announces_ready(self):
        output = queue.Queue()
        ready, done, cancel = threading.Event(), threading.Event(), threading.Event()
        model = Mock()
        model.feature_extractor.return_value = np.zeros((80, 3000), np.float32)
        model.encode.side_effect = RuntimeError('cudnn component missing')
        with patch('gpu_runtime.prepare_cuda_runtime'), \
             patch.dict('sys.modules', {'faster_whisper': SimpleNamespace(WhisperModel=lambda *a, **kw: model)}):
            _transcription_worker({}, {'device': 'cuda'}, output, ready, done, cancel)
        kind, message = output.get_nowait()
        self.assertEqual(kind, 'error')
        self.assertIn('cudnn component missing', message)
        self.assertFalse(ready.is_set())

    def test_cpu_load_does_not_require_cuda(self):
        with patch('gpu_runtime.prepare_cuda_runtime') as prepare, \
             patch.dict('sys.modules', {'faster_whisper': SimpleNamespace(WhisperModel=Mock())}):
            Transcriber(device='cpu').load()
        prepare.assert_not_called()

    def test_missing_dll_has_actionable_error(self):
        with patch.object(gpu_runtime, '_PREPARED', False), \
             patch.object(gpu_runtime, '_DLL_DIRECTORIES', []), \
             patch.object(gpu_runtime, '_LOADED_LIBRARIES', []), \
             patch('gpu_runtime.cuda_directories', return_value=[]), \
             patch('gpu_runtime.ctypes.WinDLL', side_effect=OSError(126, 'not found')), \
             patch.dict('os.environ'):
            with self.assertRaisesRegex(RuntimeError, 'setup-gpu.bat'):
                gpu_runtime.prepare_cuda_runtime()
