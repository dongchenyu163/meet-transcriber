from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from audio import MicrophoneRecorder,ContinuousResampler,AudioStore,_capture_worker
from soundcheck import test_signal,signal_correlation


class CaptureRegressionTests(unittest.TestCase):
    def test_callbacks_preserve_every_frame_across_different_read_sizes(self):
        recorder=MicrophoneRecorder()
        original=np.arange(12000*2,dtype=np.float32).reshape(-1,2)
        for start in range(0,len(original),480):
            block=original[start:start+480].copy()
            recorder.callback(block,len(block),None,False)
            block.fill(-1)
        actual=np.concatenate([recorder.record(137),recorder.record(6000),recorder.record(5863)])
        np.testing.assert_array_equal(actual,original)

    def test_callback_backpressure_reports_loss(self):
        recorder=MicrophoneRecorder()
        recorder.blocks=queue.Queue(maxsize=1)
        block=np.ones((480,1),np.float32)
        recorder.callback(block,480,None,False)
        recorder.callback(block,480,None,False)
        with self.assertRaisesRegex(RuntimeError,'溢出'):recorder.record(480)

    def test_streaming_resampler_matches_continuous_resampling(self):
        for rate in (44100,48000):
            audio=np.sin(2*np.pi*1234*np.arange(rate*2)/rate).astype(np.float32)*.5
            whole=ContinuousResampler(rate)
            expected=np.concatenate([whole.process(audio),whole.process()])
            streaming=ContinuousResampler(rate)
            pieces=[streaming.process(audio[start:start+1379]) for start in range(0,len(audio),1379)]
            actual=np.concatenate(pieces+[streaming.process()])
            self.assertEqual(len(actual),32000)
            np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_half_speed_device_is_detected_even_without_overflow_flag(self):
        class Recorder:
            def record(self,numframes):return np.ones((numframes,2),np.float32)*.1
            def __exit__(self,*args):pass
        with tempfile.TemporaryDirectory(prefix='capture-regression-') as folder:
            store=AudioStore(Path(folder)/'test.wav')
            errors=queue.Queue()
            with patch('audio._open_stream',return_value=(Recorder(),16000)),patch('audio.time.monotonic',side_effect=[0,10]):
                _capture_worker(store,None,False,threading.Event(),threading.Event(),errors)
            self.assertIn('录音时钟异常',errors.get_nowait())

    def test_loopback_check_distinguishes_test_signal_from_silence(self):
        reference=test_signal()
        recorded=np.concatenate([np.zeros(1200,np.float32),reference*.5,np.zeros(1000,np.float32)])
        self.assertGreater(signal_correlation(recorded),.99)
        self.assertEqual(signal_correlation(np.zeros_like(recorded)),0)
        random=np.random.default_rng(1).normal(0,.1,len(recorded))
        self.assertLess(signal_correlation(random),.1)


if __name__=='__main__':unittest.main()
