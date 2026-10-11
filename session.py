"""Session coordinator. Qt never records audio, loads models, or waits on I/O."""
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
import uuid

from audio import AudioStore, AudioCapture
from screenshot import Screenshotter
from storage import TranscriptWriter, recover_exports, SOURCE_LABELS
from transcribe import TranscriptionProcess
from speakers import SpeakerProcess


@dataclass
class SessionConfig:
    out_dir: str = 'sessions'
    system_device: str | None = None
    mic_device: str | None = None
    record_system: bool = True
    record_mic: bool = True
    model_size: str = 'small'
    language: str | None = None
    device: str = 'cpu'
    compute_type: str = 'int8'
    cpu_threads: int = 4
    prompt: str = ''
    window_id: int | None = None
    full_screen: bool = False
    shot_interval: float = 0
    diarize: bool = False
    participants: int = 0
    mic_is_participant: bool = True
    asr_backend: str = 'whisper'
    multilingual: bool = False
    sensevoice_variant: str = 'small-int8'

    def speaker_count(self):
        if not 0 <= self.participants <= 100:
            raise ValueError('开会人数必须在 0～100 之间。')
        if not self.participants:
            return 0
        remaining = self.participants - int(self.record_mic and self.mic_is_participant)
        if self.record_system and remaining < 1:
            raise ValueError('启用会议播放声且麦克风单独占一人时，总人数至少为 2；单人录音请关闭会议播放声。')
        return max(1, remaining)

    def model_options(self):
        return dict(model_size=self.model_size, language=self.language, device=self.device,
                    compute_type=self.compute_type, cpu_threads=self.cpu_threads,
                    initial_prompt=self.prompt, backend=self.asr_backend, multilingual=self.multilingual,
                    sensevoice_variant=self.sensevoice_variant)


class MeetingSession:
    def __init__(self, config, emit=None):
        self.config = config
        self.emit = emit or (lambda kind, value: None)
        self.stop_event = threading.Event()
        self.skip_transcription = threading.Event()
        self.shots = queue.Queue(maxsize=10)
        self.root = None
        self.recording = False
        self.cursors = {}
        self.errors = []
        self.speaker_cursor = 0
        self.speaker_done = False
        self.speaker_failed = False

    def _speaker_messages(self, speaker, writer):
        if not speaker:
            return
        for kind, value in speaker.drain():
            if kind == 'speakers':
                writer.append('speakers', value)
                self.emit('transcript_refresh', str(writer.root))
            elif kind == 'speaker_cursor':
                self.speaker_cursor = value
                self.emit('speaker_progress', value / 16000)
            elif kind == 'speaker_done':
                self.speaker_done = True
            elif kind == 'speaker_error':
                self.speaker_failed = True
                writer.append('state', {'speaker_status': 'failed'})
                self.emit('warning', value + '；录音和文字转录继续。')
        if (speaker.process.pid and not speaker.process.is_alive() and not self.speaker_done
                and not self.speaker_failed and not self.skip_transcription.is_set()):
            self.speaker_failed = True
            writer.append('state', {'speaker_status': 'failed'})
            self.emit('warning', '说话人分析进程意外退出；录音和文字转录继续。')

    def stop(self, save_only=False):
        if save_only:
            self.skip_transcription.set()
        self.stop_event.set()

    def request_screenshot(self):
        if not self.recording:
            return
        try:
            self.shots.put_nowait(True)
        except queue.Full:
            self.emit('warning', '截图请求过多，请稍候。')

    def _messages(self, asr, writer):
        for kind, value in asr.drain():
            if kind == 'line':
                writer.on_line(value)
                self.emit('line', value)
            elif kind == 'cursor':
                self.cursors[value[0]] = value[1]
            elif kind == 'error':
                raise RuntimeError(value)

    def _error(self, writer, message):
        self.errors.append(message)
        self.emit('error', message)
        if writer:
            writer.append('error', message)

    def run(self):
        config = self.config
        if not config.record_system and not config.record_mic:
            raise ValueError('请至少选择一路音频。')
        count = config.speaker_count()
        self.root = Path(config.out_dir).resolve() / (datetime.now().strftime('%Y-%m-%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6])
        self.root.mkdir(parents=True)
        self.emit('session', str(self.root))
        writer, asr, captures, speaker = None, None, {}, None
        status = 'complete'
        ended_at = time.time()
        stores = {}
        try:
            writer = TranscriptWriter(self.root, time.time(), {'model': config.model_size if config.asr_backend == 'whisper' else 'sensevoice-' + config.sensevoice_variant,
                                       'device': config.device,
                                       'asr_backend': config.asr_backend, 'multilingual': config.multilingual,
                                       'language': config.language, 'audio_files': {},
                                       'diarize': config.diarize and config.record_system,
                                       'participants': config.participants, 'system_speaker_count': count,
                                       'speaker_status': 'pending' if config.diarize and config.record_system else 'disabled'})
            for source, enabled, device, loopback in [
                ('system', config.record_system, config.system_device, True),
                ('mic', config.record_mic, config.mic_device, False)]:
                if enabled:
                    stores[source] = AudioStore(self.root / f'{source}.wav')
                    captures[source] = AudioCapture(stores[source], device, loopback)
            writer.append('state', {'audio_files': {s: f'{s}.wav' for s in stores}})
            asr = TranscriptionProcess(stores, config.model_options())
            if config.diarize and 'system' in stores:
                speaker = SpeakerProcess(stores['system'], count)
                speaker.start()
            self.emit('status', '正在加载本地转录与说话人模型，尚未开始录音…')
            asr.start()
            while not asr.ready.wait(.15):
                self._messages(asr, writer)
                self._speaker_messages(speaker, writer)
                if self.stop_event.is_set():
                    status = 'partial'
                    return str(self.root)
                if not asr.process.is_alive():
                    self._messages(asr, writer)
                    raise RuntimeError('模型进程已退出，请检查模型或运行设备设置。')
            if self.stop_event.is_set():
                status = 'partial'
                return str(self.root)
            shotter = Screenshotter(self.root / 'screenshots', config.window_id, config.full_screen)
            for capture in captures.values():
                capture.start()
            opening_at = time.monotonic()
            announced = set()
            last_samples = {source: (0, opening_at) for source in stores}
            next_shot = time.monotonic() + config.shot_interval
            self.emit('status', '正在打开音频设备…')
            while not self.stop_event.wait(.15):
                self._messages(asr, writer)
                self._speaker_messages(speaker, writer)
                if not asr.process.is_alive():
                    self._messages(asr, writer)
                    raise RuntimeError('转录进程意外退出，正在保留音频。')
                metrics = {}
                for source, capture in captures.items():
                    error = capture.poll_error()
                    if error:
                        raise RuntimeError(f'{SOURCE_LABELS[source]}：{error}')
                    if not capture.ready.is_set() and time.monotonic() - opening_at > 20:
                        raise RuntimeError(f'{SOURCE_LABELS[source]}：打开设备超时。')
                    if capture.ready.is_set() and source not in announced:
                        writer.append('audio_start', {'source': source, 'timestamp': stores[source].started_at})
                        announced.add(source)
                    count, epoch, level = stores[source].snapshot()
                    old_count, updated = last_samples[source]
                    if count != old_count:
                        last_samples[source] = (count, time.monotonic())
                    elif capture.ready.is_set() and time.monotonic() - updated > 20:
                        raise RuntimeError(f'{SOURCE_LABELS[source]}：20 秒未收到音频，设备可能已断开。')
                    metrics[source] = {'level': level, 'seconds': count / stores[source].rate,
                        'backlog': max(0, count - self.cursors.get(source, 0)) / stores[source].rate}
                if len(announced) == len(captures) and not self.recording:
                    self.recording = True
                    writer.append('state', {'status': 'recording'})
                    self.emit('status', '正在录音 · 两路音频分别保存' if len(captures) == 2 else '正在录音')
                    self.emit('recording', True)
                self.emit('metrics', metrics)
                take_shot = False
                try:
                    self.shots.get_nowait()
                    take_shot = True
                except queue.Empty:
                    pass
                if self.recording and config.shot_interval > 0 and time.monotonic() >= next_shot:
                    take_shot = True
                    next_shot = time.monotonic() + config.shot_interval
                if take_shot and self.recording:
                    try:
                        result = shotter.capture()
                        writer.add_screenshot_event(result)
                        self.emit('screenshot', result)
                    except Exception as exc:
                        self.emit('warning', f'截图失败：{exc}')
        except Exception as exc:
            status = 'failed'
            try:
                self._error(writer, str(exc))
            except Exception:
                self.emit('error', '保存记录失败，请检查输出目录和剩余磁盘空间。')
        finally:
            self.recording = False
            self.emit('recording', False)
            self.emit('status', '正在停止录音并处理剩余音频…')
            for capture in captures.values():
                capture.stop()
            for capture in captures.values():
                capture.join()
                if capture.error:
                    status = 'failed'
                    self.emit('error', capture.error)
                    if writer:
                        try:
                            writer.append('error', capture.error)
                        except OSError:
                            pass
                if capture.forced:
                    self.skip_transcription.set()
            ended_at = time.time()
            if speaker:
                speaker.capture_done.set()
            if asr is not None and asr.process.pid:
                if self.skip_transcription.is_set() or not asr.ready.is_set():
                    status = status if status == 'failed' else 'partial'
                    asr.terminate()
                else:
                    asr.capture_done.set()
                    terminated = False
                    try:
                        while asr.process.is_alive():
                            self._messages(asr, writer)
                            self._speaker_messages(speaker, writer)
                            if self.skip_transcription.wait(.15):
                                status = status if status == 'failed' else 'partial'
                                asr.terminate()
                                terminated = True
                                break
                            self.emit('status', '正在补完转录；可选“仅保存录音”稍后恢复…')
                        if not terminated:
                            self._messages(asr, writer)
                        if asr.process.exitcode not in (0, None) and status == 'complete':
                            status = 'failed'
                    except Exception as exc:
                        status = 'failed'
                        self.emit('error', str(exc))
                        asr.terminate()
                asr.close()
            if speaker:
                try:
                    while speaker.process.is_alive():
                        self._speaker_messages(speaker, writer)
                        if self.skip_transcription.wait(.15) or not asr or not asr.ready.is_set():
                            speaker.terminate()
                            break
                        self.emit('status', '正在补完说话人标注；可选“仅保存录音”取消…')
                    self._speaker_messages(speaker, writer)
                    writer.append('state', {'speaker_status': 'complete' if self.speaker_done else
                                           ('failed' if self.speaker_failed else 'partial')})
                finally:
                    speaker.terminate()
                    speaker.close()
            if writer:
                try:
                    writer.finalize(ended_at, status)
                except Exception as exc:
                    self.emit('error', f'最终导出失败；音频和事件记录可用于恢复：{exc}')
            self.emit('status', {'complete': '会议已保存', 'partial': '音频已保存，转录可稍后恢复',
                                 'failed': '录制出现错误，已保留可恢复的数据'}[status])
        return str(self.root)


def retranscribe(session_dir, config, emit, cancel):
    """Write a fresh transcript in a new subdirectory, preserving the original."""
    root = Path(session_dir).resolve()
    data = recover_exports(root)
    audio_root = Path(data.get('original_session', root)).resolve()
    if not root.is_relative_to(audio_root):
        raise ValueError('原始音频目录必须是当前恢复目录的父目录。')
    starts = data.get('audio_starts', {})
    stores = {}
    for source in ('system', 'mic'):
        path = audio_root / f'{source}.wav'
        if path.exists():
            timestamp = starts.get(source, data['session_start'])
            if path.with_suffix('.start').exists():
                timestamp = float(path.with_suffix('.start').read_text(encoding='utf-8'))
            stores[source] = AudioStore(path, started_at=timestamp, existing=True)
            starts[source] = timestamp
    if not stores:
        raise ValueError('所选会议目录中没有可恢复的 WAV。')
    # Recovery reads the saved tracks, independently of live device checkboxes.
    config = replace(config, record_system='system' in stores, record_mic='mic' in stores)
    count = config.speaker_count()
    destination = root / ('recovered_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir()
    writer = TranscriptWriter(destination, data['session_start'],
                               {'original_session': str(audio_root), 'audio_starts': starts,
                                'asr_backend': config.asr_backend, 'multilingual': config.multilingual,
                                'model': config.model_size if config.asr_backend == 'whisper' else 'sensevoice-' + config.sensevoice_variant,
                                'device': config.device,
                                'audio_files': {s: str(store.path) for s, store in stores.items()},
                                'diarize': config.diarize and 'system' in stores,
                                'participants': config.participants, 'system_speaker_count': count,
                                'speaker_status': 'pending' if config.diarize and 'system' in stores else 'disabled'})
    for shot in data.get('screenshots', []):
        writer.append('screenshot', shot)
    asr = TranscriptionProcess(stores, config.model_options())
    speaker = SpeakerProcess(stores['system'], count) if config.diarize and 'system' in stores else None
    coordinator = MeetingSession(config, emit)
    coordinator.skip_transcription = cancel
    status = 'complete'
    try:
        emit('session', str(destination))
        asr.start()
        if speaker:
            speaker.start()
            speaker.capture_done.set()
        asr.capture_done.set()
        emit('status', '正在加载模型并重新转录完整音频…')
        while True:
            coordinator._speaker_messages(speaker, writer)
            for kind, value in asr.drain():
                if kind == 'line':
                    writer.on_line(value)
                    emit('line', value)
                elif kind == 'error':
                    raise RuntimeError(value)
            if not asr.process.is_alive():
                # The child flushes the queue before process exit.
                for kind, value in asr.drain():
                    if kind == 'line':
                        writer.on_line(value)
                        emit('line', value)
                    elif kind == 'error':
                        raise RuntimeError(value)
                if asr.process.exitcode != 0:
                    raise RuntimeError('恢复转录进程异常退出。')
                if not speaker or not speaker.process.is_alive():
                    coordinator._speaker_messages(speaker, writer)
                    break
            if cancel.wait(.15):
                status = 'partial'
                asr.terminate()
                if speaker:
                    speaker.terminate()
                break
    except Exception:
        status = 'failed'
        asr.terminate()
        if speaker:
            speaker.terminate()
        raise
    finally:
        asr.close()
        if speaker:
            coordinator._speaker_messages(speaker, writer)
            speaker.terminate()
            speaker.close()
            writer.append('state', {'speaker_status': 'complete' if coordinator.speaker_done else
                                   ('failed' if coordinator.speaker_failed else 'partial')})
        writer.finalize(data.get('session_end') or time.time(), status)
    return str(destination)
