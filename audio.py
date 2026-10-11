"""Two independent WASAPI capture processes and a durable, seekable WAV store."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
import multiprocessing as mp
import os
from pathlib import Path
import queue
import struct
import time
import wave
import numpy as np

RATE = 16000
CONTEXT = mp.get_context('spawn')


@dataclass(frozen=True)
class AudioDevice:
    id: str
    name: str
    loopback: bool


def list_devices():
    import soundcard as sc
    return [AudioDevice(m.id, m.name, bool(m.isloopback))
            for m in sc.all_microphones(include_loopback=True)]


def describe_error(exc):
    """Assertions and some native exceptions have an empty str()."""
    message = str(exc).strip()
    return f'{type(exc).__name__}: {message}' if message else type(exc).__name__


class MicrophoneRecorder:
    """Drain continuous WASAPI callbacks; never use PortAudio's blocking read.

    Some Windows devices (including C920) silently lose alternating packets in
    PortAudio's blocking WASAPI path, even without an overflow flag.
    """
    def __init__(self, stream=None):
        self.stream = stream
        self.blocks = queue.Queue(maxsize=512)
        self.pending = None
        self.error = None

    def callback(self, data, frames, timing, status):
        if status:
            self.error = '麦克风采集出现丢帧：' + str(status)
            return
        try:
            self.blocks.put_nowait(data.copy())
        except queue.Full:
            self.error = '麦克风缓冲溢出，磁盘或系统处理跟不上采集。'

    def record(self, numframes):
        chunks, remaining = [], numframes
        deadline = time.monotonic() + 5
        while remaining:
            if self.error:
                raise RuntimeError(self.error)
            if self.pending is None or not len(self.pending):
                try:
                    self.pending = self.blocks.get(timeout=.1)
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('麦克风 5 秒没有送来音频，请检查设备。')
                    continue
            count = min(remaining, len(self.pending))
            chunks.append(self.pending[:count])
            self.pending = self.pending[count:]
            remaining -= count
        return np.concatenate(chunks, axis=0)

    def __exit__(self, *args):
        try:
            self.stream.stop()
        finally:
            self.stream.close()


def _open_microphone(mic):
    # SoundCard 0.4.2 asserts WAVE_FORMAT_EXTENSIBLE, but webcams such as the
    # C920 expose ordinary PCM. PortAudio handles PCM and native channel counts.
    import sounddevice as sd
    devices = sd.query_devices()
    hosts = sd.query_hostapis()
    matches = [(index, info) for index, info in enumerate(devices)
               if info['max_input_channels'] > 0
               and hosts[info['hostapi']]['name'] == 'Windows WASAPI'
               and info['name'].casefold() == mic.name.casefold()]
    if len(matches) != 1:
        raise RuntimeError(f'无法唯一匹配 WASAPI 麦克风“{mic.name}”，请刷新并重新选择设备。')
    index, info = matches[0]
    rates = list(dict.fromkeys([int(info['default_samplerate']), RATE, 48000, 44100]))
    channels = min(2, int(info['max_input_channels']))
    errors = []
    for rate in rates:
        stream = None
        try:
            recorder = MicrophoneRecorder()
            stream = sd.InputStream(device=index, samplerate=rate, channels=channels,
                                    dtype='float32', latency='high', callback=recorder.callback)
            recorder.stream = stream
            stream.start()
            return recorder, rate
        except Exception as exc:
            errors.append(f'{rate}Hz/{channels}ch: {describe_error(exc)}')
            if stream is not None:
                stream.close()
    raise RuntimeError(f'无法打开麦克风“{mic.name}”：' + '; '.join(errors))


class AudioStore:
    """Mono PCM16 WAV with one process lock for data, count and timestamp.

    Audio never expires. Header and data are flushed each block. Recovery can
    reconstruct the header from file length after an interrupted write.
    """
    def __init__(self, path, rate=RATE, started_at=None, existing=False):
        self.path = Path(path)
        self.rate = rate
        self.lock = CONTEXT.RLock()
        self.count = CONTEXT.Value('q', 0, lock=False)
        self.epoch = CONTEXT.Value('d', started_at or time.time(), lock=False)
        self.level = CONTEXT.Value('d', 0, lock=False)
        if existing:
            self.repair_header(self.path)
            with wave.open(str(self.path), 'rb') as f:
                if f.getframerate() != rate or f.getnchannels() != 1 or f.getsampwidth() != 2:
                    raise ValueError('仅支持本程序生成的 16kHz 单声道 PCM16 WAV。')
                self.count.value = f.getnframes()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(self.path), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(rate)
                f.writeframes(b'')

    @staticmethod
    def repair_header(path):
        path = Path(path)
        with path.open('r+b') as f:
            header = f.read(44)
            if len(header) != 44 or header[:4] != b'RIFF' or header[36:40] != b'data':
                raise ValueError('不是本程序的 WAV 文件。')
            size = path.stat().st_size - 44
            size -= size % 2
            f.truncate(size + 44)
            f.seek(4)
            f.write(struct.pack('<I', size + 36))
            f.seek(40)
            f.write(struct.pack('<I', size))
            f.flush()
            os.fsync(f.fileno())

    @contextmanager
    def _locked(self):
        if not self.lock.acquire(timeout=3):
            raise RuntimeError('音频存储锁超时，采集可能中断；请恢复保存的 WAV。')
        try:
            yield
        finally:
            self.lock.release()

    @property
    def started_at(self):
        with self._locked():
            return self.epoch.value

    def set_started_at(self, timestamp):
        with self._locked():
            if self.count.value:
                raise RuntimeError('音频开始后不可改变时间原点。')
            self.epoch.value = timestamp
            from storage import atomic_write
            atomic_write(self.path.with_suffix('.start'), str(timestamp))

    def push(self, samples):
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if not samples.size:
            return
        data = (np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()
        with self._locked():
            size = (self.count.value + samples.size) * 2
            if size + 36 >= 2**32:
                raise RuntimeError('WAV 已达文件大小上限，请开始新会议。')
            with self.path.open('r+b') as f:
                f.seek(44 + self.count.value * 2)
                f.write(data)
                f.seek(4)
                f.write(struct.pack('<I', size + 36))
                f.seek(40)
                f.write(struct.pack('<I', size))
                f.flush()
                os.fsync(f.fileno())
            self.count.value += samples.size
            self.level.value = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    def snapshot(self):
        with self._locked():
            return self.count.value, self.epoch.value, self.level.value

    def read(self, start, frames):
        with self._locked():
            count = max(0, min(frames, self.count.value - start))
            epoch = self.epoch.value
            with self.path.open('rb') as f:
                f.seek(44 + max(0, start) * 2)
                data = f.read(count * 2)
            return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768, epoch


def _open_stream(device_id, loopback):
    import soundcard as sc
    devices = [m for m in sc.all_microphones(include_loopback=True)
               if bool(m.isloopback) == loopback]
    if device_id:
        devices = [m for m in devices if m.id == device_id]
    else:
        default = sc.default_speaker() if loopback else sc.default_microphone()
        devices = [m for m in devices if m.id == default.id]
    if not devices:
        raise RuntimeError('所选音频设备不存在，请刷新设备列表。')
    mic = devices[0]
    if not loopback:
        return _open_microphone(mic)
    errors = []
    for rate in (RATE, 48000, 44100):
        try:
            # WASAPI can remix; avoid SoundCard's Windows single-channel issue.
            recorder = mic.recorder(samplerate=rate, channels=max(2, int(mic.channels)))
            recorder.__enter__()
            return recorder, rate
        except Exception as exc:
            errors.append(f'{rate}Hz: {describe_error(exc)}')
    raise RuntimeError(f'无法打开播放声设备“{mic.name}”：' + '; '.join(errors))


class ContinuousResampler:
    """Keep filter history and fractional sample positions between audio blocks."""
    def __init__(self, source_rate, target_rate=RATE):
        import av
        self.av = av
        self.source_rate, self.target_rate = source_rate, target_rate
        self.resampler = av.AudioResampler(format='flt', layout='mono', rate=target_rate)

    def process(self, samples=None):
        frame = None
        if samples is not None:
            frame = self.av.AudioFrame.from_ndarray(np.ascontiguousarray(samples.reshape(1,-1),dtype=np.float32),
                                                  format='flt', layout='mono')
            frame.sample_rate = self.source_rate
        frames = self.resampler.resample(frame)
        return np.concatenate([f.to_ndarray().ravel() for f in frames]) if frames else np.empty(0,np.float32)


def _capture_worker(store, device_id, loopback, stop, ready, errors):
    recorder = None
    try:
        recorder, rate = _open_stream(device_id, loopback)
        frames = rate // 10
        resampler = ContinuousResampler(rate, store.rate) if rate != store.rate else None
        store.set_started_at(time.time())
        began = time.monotonic()
        input_frames = 0
        ready.set()
        while not stop.is_set():
            block = recorder.record(numframes=frames)
            mono = np.asarray(block, dtype=np.float32).mean(axis=1)
            input_frames += len(mono)
            if resampler:
                mono = resampler.process(mono)
            store.push(mono)
            elapsed = time.monotonic() - began
            if elapsed >= 8 and input_frames / rate < elapsed * .85:
                raise RuntimeError(f'录音时钟异常：经过 {elapsed:.1f} 秒仅收到 {input_frames/rate:.1f} 秒音频；'
                                   '已停止采集，防止保存时长错误的录音。')
        if resampler:
            store.push(resampler.process())
    except Exception as exc:
        errors.put(describe_error(exc))
    finally:
        if recorder is not None:
            try:
                recorder.__exit__(None, None, None)
            except Exception as exc:
                errors.put(describe_error(exc))


class AudioCapture:
    """A stalled device can be stopped without hanging the Qt event loop."""
    def __init__(self, store, device_id=None, loopback=True):
        self.store = store
        self.stop_event = CONTEXT.Event()
        self.ready = CONTEXT.Event()
        self.errors = CONTEXT.Queue()
        self.error = None
        self.forced = False
        self.process = CONTEXT.Process(target=_capture_worker,
            args=(store, device_id, loopback, self.stop_event, self.ready, self.errors),
            name='SystemAudio' if loopback else 'Microphone', daemon=True)

    def start(self):
        self.process.start()

    def poll_error(self):
        try:
            self.error = self.errors.get_nowait()
        except queue.Empty:
            pass
        if self.process.pid and not self.process.is_alive() and not self.stop_event.is_set():
            self.error = self.error or f'录音进程提前退出（{self.process.exitcode}）。'
        return self.error

    def stop(self):
        self.stop_event.set()

    def join(self):
        if not self.process.pid:
            return
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.forced = True
            self.error = '音频设备停止超时，已终止采集；请恢复保存的 WAV。'
        self.poll_error()
        self.errors.close()
        self.errors.join_thread()
