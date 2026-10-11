"""Bounded-window, local speaker diarization alongside the Whisper process."""
import json
from pathlib import Path
import queue
import numpy as np

from audio import CONTEXT, RATE, describe_error
from speaker_models import SEGMENTATION, EMBEDDING, available
from settings import ROOT
from storage import atomic_write

PROFILES = ROOT / 'speaker-profiles.json'


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm < 1e-6:
        raise ValueError('无效的声纹。')
    return vector / norm


class SpeakerEngine:
    def __init__(self, threads=2, cluster_threshold=.5):
        if not available():
            raise RuntimeError('请在“说话人”设置中先下载分析模型。')
        import sherpa_onnx as so
        embedding = so.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING),
                         num_threads=threads, provider='cpu')
        segmentation = so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEGMENTATION), window_shift_ratio=.1),
            num_threads=threads, provider='cpu')
        self.pipeline = so.OfflineSpeakerDiarization(so.OfflineSpeakerDiarizationConfig(
            segmentation=segmentation, embedding=embedding,
            clustering=so.FastClusteringConfig(num_clusters=-1, threshold=cluster_threshold),
            min_duration_on=.3, min_duration_off=.5))
        self.extractor = so.SpeakerEmbeddingExtractor(embedding)

    def embed(self, samples):
        stream = self.extractor.create_stream()
        stream.accept_waveform(RATE, np.ascontiguousarray(samples, dtype=np.float32))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        return unit(self.extractor.compute(stream))

    def analyze(self, samples, independent=False):
        if len(samples) < RATE * .5 or np.sqrt(np.mean(samples ** 2)) < .0001:
            return [], {}
        turns = [{'start': float(s.start), 'end': float(s.end), 'cluster': int(s.speaker)}
                 for s in self.pipeline.process(samples).sort_by_start_time()]
        vectors = {}
        for cluster in sorted({t['cluster'] for t in turns}):
            clips = []
            for turn in turns:
                if turn['cluster'] != cluster or turn['end'] - turn['start'] < .5:
                    continue
                # Overlapping voices are not suitable enrollment material.
                if any(other['cluster'] != cluster and
                       min(turn['end'], other['end']) > max(turn['start'], other['start'])
                       for other in turns):
                    continue
                clips.append(samples[round(turn['start'] * RATE):round(turn['end'] * RATE)])
            if clips:
                vector = self.embed(np.concatenate(clips)[:RATE * 15])
                if vector is not None:
                    vectors[cluster] = vector
        if independent:
            return self._independent_turns(samples, turns, vectors)
        return turns, vectors

    def _independent_turns(self, samples, turns, fallback):
        """Keep separate utterance features so global clustering can undo merges."""
        pieces, vectors = [], {}
        for turn in turns:
            others = [other for other in turns if other['cluster'] != turn['cluster']
                      and min(turn['end'], other['end']) > max(turn['start'], other['start'])]
            edges = sorted({turn['start'], turn['end']} |
                           {max(turn['start'], min(turn['end'], t[key]))
                            for t in others for key in ('start', 'end')})
            for left, right in zip(edges, edges[1:]):
                overlap = any(min(right, t['end']) > max(left, t['start']) for t in others)
                # Even long uninterrupted speech gets multiple voice samples.
                bounds = np.linspace(left, right, max(1, int(np.ceil((right-left)/4)))+1)
                for start, end in zip(bounds, bounds[1:]):
                    index = len(pieces)
                    pieces.append({'start': float(start), 'end': float(end),
                                   'cluster': index, 'overlap': overlap})
                    if not overlap:
                        vector = self.embed(samples[round(start*RATE):round(end*RATE)]) if end-start >= .7 else None
                        vector = vector if vector is not None else fallback.get(turn['cluster'])
                        if vector is not None:
                            vectors[index] = vector
        return pieces, vectors


class SpeakerTracker:
    """Session IDs persist across windows; names require enrolled voice profiles."""
    def __init__(self, profiles=None, match_threshold=.6, margin=.08):
        self.centers = {}
        self.next_id = 1
        self.profiles = profiles or {}
        self.match_threshold, self.margin = match_threshold, margin

    def assign(self, vectors, cannot_link=()):
        # The window's clustering can split one voice into several clusters.
        # Merge matching centroids, except voices actually speaking simultaneously.
        assigned = {}
        for cluster, vector in vectors.items():
            vector = unit(vector)
            forbidden = {assigned[other] for a, b in cannot_link for other in
                         ([b] if a == cluster else ([a] if b == cluster else [])) if other in assigned}
            scores = sorted([(float(np.dot(vector, center)), speaker)
                             for speaker, center in self.centers.items() if speaker not in forbidden], reverse=True)
            if not scores or scores[0][0] < self.match_threshold:
                speaker = f'SPEAKER_{self.next_id:02d}'
                self.next_id += 1
                assigned[cluster] = speaker
                self.centers[speaker] = vector
            else:
                speaker = scores[0][1]
                assigned[cluster] = speaker
                self.centers[speaker] = unit(self.centers[speaker] * .8 + vector * .2)
        result = {}
        for cluster, speaker in assigned.items():
            if speaker == 'UNKNOWN':
                result[cluster] = {'speaker': 'UNKNOWN', 'speaker_name': None}
                continue
            scores = sorted([(float(np.dot(unit(vectors[cluster]), unit(profile))), name)
                             for name, profile in self.profiles.items()], reverse=True)
            name = None
            if scores and scores[0][0] >= self.match_threshold and (
                    len(scores) == 1 or scores[0][0] - scores[1][0] >= self.margin):
                name = scores[0][1]
            result[cluster] = {'speaker': speaker, 'speaker_name': name}
        return result


def load_profiles(path=PROFILES):
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('model') != '3dspeaker-eres2net-base-zh-16k':
        raise ValueError('声音样本模型版本不匹配，请重新登记。')
    return {name: unit(value) for name, value in data['profiles'].items()}


def enroll(name, files, progress=lambda _: None, cancel=None):
    name = name.strip()
    if not name or len(name) > 80 or '\n' in name:
        raise ValueError('请填写 1～80 字的姓名。')
    from faster_whisper.audio import decode_audio
    engine = SpeakerEngine()
    vectors = []
    seconds = 0
    for filename in files:
        if cancel and cancel.is_set():
            raise RuntimeError('已取消声音登记。')
        progress('正在登记声音样本：' + Path(filename).name)
        samples = decode_audio(str(filename), sampling_rate=RATE)
        # Bounded inference even if a long sample is selected.
        for start in range(0, min(len(samples), RATE * 120), RATE * 20):
            if cancel and cancel.is_set():
                raise RuntimeError('已取消声音登记。')
            clip = samples[start:start + RATE * 20]
            turns, _ = engine.analyze(clip)
            clean = [clip[round(t['start'] * RATE):round(t['end'] * RATE)] for t in turns
                     if not any(o['cluster'] != t['cluster'] and
                         min(t['end'], o['end']) > max(t['start'], o['start']) for o in turns)]
            if clean:
                voice = np.concatenate(clean)
                if len(voice) >= RATE * 2:
                    vector = engine.embed(voice)
                    if vector is not None:
                        vectors.append(vector)
                        seconds += len(voice) / RATE
    if seconds < 5 or not vectors:
        raise ValueError('样本中有效人声不足 5 秒，请使用清晰、仅一人说话的录音。')
    profiles = load_profiles()
    if cancel and cancel.is_set():
        raise RuntimeError('已取消声音登记。')
    profiles[name] = unit(np.mean(vectors, axis=0))
    atomic_write(PROFILES, json.dumps({'model': '3dspeaker-eres2net-base-zh-16k',
        'profiles': {n: v.tolist() for n, v in profiles.items()}}, ensure_ascii=False, indent=2))
    return f'已登记 {name}（有效人声 {seconds:.0f} 秒；声纹保存在本机）'


def _speaker_worker(store, output, ready, capture_done, cancel, num_speakers=0):
    try:
        engine = SpeakerEngine()
        profiles = load_profiles()
        tracker = SpeakerTracker(profiles)
        if num_speakers:
            from speaker_clustering import MeetingClustering
            meeting = MeetingClustering(num_speakers, profiles)
        else:
            meeting = None
        ready.set()
        cursor = 0
        while not cancel.is_set():
            total, _, _ = store.snapshot()
            final = capture_done.is_set()
            if total <= cursor:
                if final:
                    if meeting:
                        revised = meeting.update(final=True)
                        if revised:
                            output.put(('speakers', {'upsert': True, 'turns': revised}))
                    break
                cancel.wait(.3)
                continue
            if not final and total - cursor < RATE * 24:
                cancel.wait(.3)
                continue
            left = max(0, cursor - RATE * 4)
            samples, epoch = store.read(left, min(RATE * 36, total - left))
            end = left + len(samples)
            boundary = end if final and end == total else end - RATE * 4
            if boundary <= cursor:
                cancel.wait(.3)
                continue
            turns, vectors = engine.analyze(samples, independent=True) if meeting else engine.analyze(samples)
            if meeting:
                observations = []
                for turn in turns:
                    start = max(cursor, left + round(turn['start'] * RATE))
                    finish = min(boundary, left + round(turn['end'] * RATE))
                    if finish > start:
                        observations.append({'start': epoch + start / RATE, 'end': epoch + finish / RATE,
                            'vector': vectors.get(turn['cluster']), 'overlap': turn.get('overlap', False)})
                revised = meeting.update(observations, final=final and boundary == total)
                output.put(('speakers', {'upsert': True, 'turns': revised}))
                cursor = boundary
                output.put(('speaker_cursor', cursor))
                continue
            cannot_link = {(a['cluster'], b['cluster']) for a in turns for b in turns
                           if a['cluster'] != b['cluster'] and min(a['end'], b['end']) > max(a['start'], b['start'])}
            labels = tracker.assign(vectors, cannot_link)
            result = []
            for turn in turns:
                start = max(cursor, left + round(turn['start'] * RATE))
                finish = min(boundary, left + round(turn['end'] * RATE))
                if finish > start:
                    result.append({'start': epoch + start / RATE, 'end': epoch + finish / RATE,
                                   **labels.get(turn['cluster'], {'speaker': 'UNKNOWN', 'speaker_name': None})})
            output.put(('speakers', {'start': epoch + cursor / RATE, 'end': epoch + boundary / RATE,
                                    'turns': result}))
            cursor = boundary
            output.put(('speaker_cursor', cursor))
        output.put(('speaker_done', None))
    except Exception as exc:
        output.put(('speaker_error', '说话人分析失败：' + describe_error(exc)))


class SpeakerProcess:
    def __init__(self, store, num_speakers=0):
        self.output = CONTEXT.Queue()
        self.ready, self.capture_done, self.cancel = (CONTEXT.Event() for _ in range(3))
        self.process = CONTEXT.Process(target=_speaker_worker,
            args=(store, self.output, self.ready, self.capture_done, self.cancel, num_speakers),
            name='Speakers', daemon=True)

    def start(self): self.process.start()

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
            self.process.join(.3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()

    def close(self):
        if self.process.pid:
            self.process.join()
        self.output.close()
        self.output.join_thread()
