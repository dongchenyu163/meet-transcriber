"""SenseVoiceSmall ONNX: local Chinese/Japanese/English transcription."""
import hashlib
import os
from pathlib import Path
import tarfile
import threading
import httpx
import numpy as np
from settings import ROOT

MODEL_DIR = ROOT / 'models' / 'sensevoice'
PACKAGE = 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17'
URL = f'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{PACKAGE}.tar.bz2'
HASHES={'model.int8.onnx':'c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51',
        'tokens.txt':'f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc'}


MODELS = {
    'small-int8': {'label': 'Small · INT8（CPU）', 'directory': '.',
        'package': PACKAGE, 'file': 'model.int8.onnx', 'hashes': HASHES,
        'archive_hash': '7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e'},
    'small-fp32': {'label': 'Small · FP32（CPU / GPU）', 'directory': 'fp32-2024-07-17',
        'package': 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17',
        'file': 'model.onnx', 'hashes': {
            'model.onnx': '977016bd9c79f9eb343430b5cc305e07ab64d5212dff41b0dcfa1694bee9a8cb',
            'tokens.txt': HASHES['tokens.txt']},
        'archive_hash': None},
}


def model_spec(variant):
    if variant not in MODELS:
        raise ValueError('未知的 SenseVoice 模型版本。')
    return MODELS[variant]


def available(variant='small-int8'):
    spec = model_spec(variant)
    root = MODEL_DIR / spec['directory']
    return all((root/name).is_file() for name in (spec['file'], 'tokens.txt'))


def file_hash(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def install(progress=print, cancel=None, variant='small-int8'):
    cancel = cancel or threading.Event()
    spec = model_spec(variant)
    root = MODEL_DIR / spec['directory']
    root.mkdir(parents=True, exist_ok=True)
    if available(variant) and spec['hashes'] and all(file_hash(root/name)==checksum for name,checksum in spec['hashes'].items()):
        progress(f"SenseVoice {spec['label']} 本地模型已就绪。")
        return
    archive_path = root/'model.download'
    url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{spec['package']}.tar.bz2"
    try:
        with httpx.stream('GET',url,follow_redirects=True,timeout=60) as response:
            response.raise_for_status()
            count, reported = 0, -1
            with archive_path.open('wb') as file:
                for chunk in response.iter_bytes(1024*1024):
                    if cancel.is_set():raise RuntimeError('已取消模型下载。')
                    file.write(chunk)
                    count += len(chunk)
                    if count//(50*1024*1024) != reported:
                        reported=count//(50*1024*1024)
                        progress(f"正在下载 SenseVoice {spec['label']}：{count//(1024*1024)} MB")
        if spec['archive_hash'] and file_hash(archive_path) != spec['archive_hash']:
            raise ValueError('模型压缩包校验失败，请重试。')
        progress('下载完成，正在解压并校验模型文件…')
        with tarfile.open(archive_path,'r:bz2') as archive:
            required = {spec['file'], 'tokens.txt'}
            optional = {'LICENSE', 'test_wavs/zh.wav', 'test_wavs/en.wav', 'test_wavs/ja.wav'}
            found = set()
            for member in archive:
                prefix = spec['package']+'/'
                if not member.name.startswith(prefix):continue
                name = member.name[len(prefix):]
                if name not in required | optional:continue
                if cancel.is_set():raise RuntimeError('已取消模型安装。')
                if not member.isfile() or not 0 < member.size < 2_000_000_000:
                    raise ValueError('模型包文件无效。')
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_suffix('.tmp')
                try:
                    with archive.extractfile(member) as source,temp.open('wb') as destination:
                        while chunk:=source.read(1024*1024):
                            if cancel.is_set():raise RuntimeError('已取消模型安装。')
                            destination.write(chunk)
                    if name in spec['hashes'] and file_hash(temp)!=spec['hashes'][name]:
                        raise ValueError('模型文件校验失败，请重试。')
                    os.replace(temp,path)
                    found.add(name)
                finally:temp.unlink(missing_ok=True)
            if not required <= found:raise ValueError('模型包缺少必要文件。')
        progress(f"SenseVoice {spec['label']} 已就绪；音频在本机处理，不上传。")
    finally:archive_path.unlink(missing_ok=True)


def timed_tokens(result, duration):
    """CTC supplies token starts; derive conservative token ends for alignment."""
    tokens, starts = list(result.tokens), list(result.timestamps)
    if len(tokens) != len(starts):
        raise RuntimeError('SenseVoice 返回的字词时间戳不完整。')
    words=[]
    for i,(token,start) in enumerate(zip(tokens,starts)):
        text=token.replace('▁',' ')
        if text.startswith('<|') and text.endswith('|>'):continue
        start=max(0,float(start))
        next_start=float(starts[i+1]) if i+1<len(starts) else duration
        end=min(duration, start+.6, next_start)
        if text and end>start:words.append((start,end,text))
    if result.text.strip() and not words:
        raise RuntimeError('SenseVoice 没有返回可用的字词时间戳。')
    return words


def speech_regions(audio):
    from faster_whisper.vad import get_speech_timestamps,VadOptions
    return get_speech_timestamps(audio,VadOptions(min_silence_duration_ms=250,
        max_speech_duration_s=8,min_speech_duration_ms=150,speech_pad_ms=120))


class SenseVoiceTranscriber:
    def __init__(self,language=None,threads=4,device='cpu',variant='small-int8'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('SenseVoice 仅支持 CPU 或 NVIDIA GPU。')
        spec = model_spec(variant)
        if device == 'cuda' and variant != 'small-fp32':
            raise ValueError('SenseVoice GPU 请选 Small · FP32；INT8 版本用于 CPU。')
        if not available(variant):
            raise RuntimeError(f"请先在“转录模型”设置中下载 SenseVoice {spec['label']}。")
        if device == 'cuda':
            from gpu_runtime import prepare_sensevoice_cuda
            prepare_sensevoice_cuda()
        import sherpa_onnx
        root = MODEL_DIR / spec['directory']
        self.recognizer=sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(root/spec['file']),tokens=str(root/'tokens.txt'),
            num_threads=threads,use_itn=True,language=language or 'auto',provider=device)
        if device == 'cuda':
            # Do a real forward pass before the worker announces readiness.
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, np.zeros(16000, np.float32))
            self.recognizer.decode_stream(stream)

    def transcribe(self,audio):
        # A single language tag for a long mixed-language clip can suppress
        # entire utterances. Detect speech boundaries and decode each separately.
        words=[]
        for segment in speech_regions(audio):
            start,end=segment['start'],segment['end']
            clip=audio[start:end]
            stream=self.recognizer.create_stream()
            stream.accept_waveform(16000,np.ascontiguousarray(clip,dtype=np.float32))
            self.recognizer.decode_stream(stream)
            words.extend((s+start/16000,e+start/16000,t) for s,e,t in timed_tokens(stream.result,len(clip)/16000))
        return words


if __name__=='__main__': install()
