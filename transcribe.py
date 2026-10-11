"""Sequential, word-aware streaming transcription with a single inference owner."""
from __future__ import annotations
from dataclasses import dataclass, field
import queue
import time
import numpy as np
from audio import CONTEXT, RATE


@dataclass
class TranscriptLine:
    start: float
    end: float
    text: str
    source: str = 'system'
    screenshots: list[str] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)


class Transcriber:
    def __init__(self, model_size='small', language=None, device='cpu',
                 compute_type='int8', cpu_threads=4, initial_prompt='', backend='whisper', multilingual=False,
                 sensevoice_variant='small-int8'):
        self.options = dict(model_size=model_size, language=language, device=device,
                            compute_type=compute_type, cpu_threads=cpu_threads,
                            initial_prompt=initial_prompt, backend=backend, multilingual=multilingual,
                            sensevoice_variant=sensevoice_variant)
        self.model = None

    def load(self):
        opt = self.options
        if opt['backend'] == 'sensevoice':
            from local_asr import SenseVoiceTranscriber
            self.model = SenseVoiceTranscriber(opt['language'], opt['cpu_threads'], opt['device'], opt['sensevoice_variant'])
            return
        if opt['backend'] != 'whisper':
            raise ValueError('未知的本地转录模型。')
        if opt['device'] == 'cuda':
            from gpu_runtime import prepare_cuda_runtime
            prepare_cuda_runtime()
        from faster_whisper import WhisperModel
        self.model = WhisperModel(opt['model_size'], device=opt['device'],
            compute_type=opt['compute_type'], cpu_threads=opt['cpu_threads'])
        if opt['device'] == 'cuda':
            # Whisper defers cuBLAS/cuDNN loading until its first forward pass.
            # Run the encoder before announcing ready or opening audio devices.
            features = self.model.feature_extractor(np.zeros(RATE, dtype=np.float32))
            self.model.encode(features)

    def transcribe(self, audio):
        if self.model is None:
            raise RuntimeError('模型尚未加载。')
        if self.options['backend'] == 'sensevoice':
            return self.model.transcribe(audio)
        mixed = self.options['multilingual']
        if mixed:
            from local_asr import speech_regions
            regions = speech_regions(audio)
        else:
            regions = [{'start': 0, 'end': len(audio)}]
        # Word boundaries let a word spanning a window be committed exactly once.
        result = []
        for region in regions:
            offset = region['start'] / RATE
            # Decode each utterance with a fresh language decision. Whisper's
            # multilingual flag alone can translate or omit a language switch.
            segments, _ = self.model.transcribe(audio[region['start']:region['end']],
                language=None if mixed else self.options['language'], beam_size=3,
                vad_filter=not mixed, vad_parameters={'min_silence_duration_ms': 400},
                word_timestamps=True, condition_on_previous_text=False,
                initial_prompt=self.options['initial_prompt'] or None)
            for seg in segments:
                words = [(offset + float(w.start), offset + float(w.end), w.word) for w in (seg.words or [])]
                if not words and seg.text.strip():
                    words = [(offset + float(seg.start), offset + float(seg.end), seg.text)]
                result.extend(words)
        return result


class StreamingTranscriber:
    """One cursor per source; retain both left context and unconfirmed right audio.

    Unlike a tail buffer, every input sample is visited even with a long backlog.
    A trailing word is held for the next window rather than irreversibly chopped.
    """
    def __init__(self, store, transcriber, on_line, source='system', min_chunk=3.0,
                 max_window=25.0, overlap=1.0, silence_rms=0.0001):
        if not (0 < min_chunk <= max_window and 0 <= overlap < max_window / 3):
            raise ValueError('音频窗口参数无效。')
        self.store, self.transcriber, self.on_line = store, transcriber, on_line
        self.source = source
        self.min_chunk = int(min_chunk * store.rate)
        self.max_window = int(max_window * store.rate)
        self.overlap = int(overlap * store.rate)
        self.silence_rms = silence_rms
        self.cursor = 0

    def step(self, final=False):
        total, _, _ = self.store.snapshot()
        pending = total - self.cursor
        if pending <= 0 or (not final and pending < self.min_chunk + self.overlap):
            return False
        left = max(0, self.cursor - self.overlap)
        audio, epoch = self.store.read(left, min(self.max_window, total - left))
        end = left + audio.size
        last_window = final and end == total
        boundary = end if last_window else end - self.overlap
        if boundary <= self.cursor:
            return False
        confirmed = boundary
        if audio.size and float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) >= self.silence_rms:
            words = self.transcriber.transcribe(audio)
            selected = []
            for start, finish, text in words:
                start = max(left, left + round(start * self.store.rate))
                finish = min(end, left + round(finish * self.store.rate))
                if finish <= self.cursor or finish <= start:
                    continue
                if finish > boundary and not last_window:
                    # Revisit this entire word with right context next time.
                    confirmed = max(self.cursor, min(confirmed, start))
                    break
                selected.append((max(start, self.cursor), finish, text))
            if selected:
                self.on_line(TranscriptLine(epoch + selected[0][0] / self.store.rate,
                    epoch + selected[-1][1] / self.store.rate,
                    ''.join(w[2] for w in selected).strip(), self.source,
                    words=[{'start': epoch + s / self.store.rate,
                            'end': epoch + e / self.store.rate, 'text': t} for s, e, t in selected]))
            if confirmed <= self.cursor:
                # A model can return one segment longer than the window. Preserve
                # it with a forced full-window commit rather than looping forever.
                if words and (final or end - left == self.max_window):
                    remainder = [(s, e, t) for s, e, t in words
                                 if left + round(e * self.store.rate) > self.cursor]
                    if remainder:
                        self.on_line(TranscriptLine(epoch + self.cursor / self.store.rate,
                            epoch + end / self.store.rate,
                            ''.join(t for _, _, t in remainder).strip(), self.source))
                    confirmed = end
                else:
                    return False
        self.cursor = confirmed
        return True

    def flush(self):
        while self.step(final=True):
            pass


def _transcription_worker(stores, model_options, output, ready, capture_done, cancel):
    try:
        model = Transcriber(**model_options)
        model.load()
        ready.set()
        processors = [StreamingTranscriber(store, model,
                       lambda line: output.put(('line', line)), source)
                      for source, store in stores.items()]
        while not cancel.is_set():
            progressed = False
            final = capture_done.is_set()
            for processor in processors:
                if cancel.is_set():
                    return
                progressed |= processor.step(final=final)
                output.put(('cursor', (processor.source, processor.cursor)))
            if final and not progressed:
                break
            if not progressed:
                cancel.wait(0.3)
        output.put(('done', None))
    except Exception as exc:
        output.put(('error', f'本地转录失败：{exc}'))


class TranscriptionProcess:
    def __init__(self, stores, options):
        self.output = CONTEXT.Queue()
        self.ready = CONTEXT.Event()
        self.capture_done = CONTEXT.Event()
        self.cancel = CONTEXT.Event()
        self.process = CONTEXT.Process(target=_transcription_worker,
            args=(stores, options, self.output, self.ready, self.capture_done, self.cancel),
            name='LocalASR', daemon=True)

    def start(self):
        self.process.start()

    def drain(self):
        result = []
        while True:
            try:
                result.append(self.output.get_nowait())
            except queue.Empty:
                return result

    def terminate(self):
        self.cancel.set()
        if self.process.pid:
            # Queue output is drained by the coordinator before/after this call.
            self.process.join(0.5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()

    def close(self):
        if self.process.pid:
            self.process.join()
        self.output.close()
        self.output.join_thread()
