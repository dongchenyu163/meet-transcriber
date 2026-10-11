"""Download the two public ONNX models from the sherpa-onnx project."""
from pathlib import Path
import hashlib
import os
import tarfile
import threading
import httpx
from settings import ROOT

MODEL_DIR = ROOT / 'models' / 'speakers'
SEGMENTATION = MODEL_DIR / 'segmentation.onnx'
EMBEDDING = MODEL_DIR / 'embedding.onnx'
RELEASES = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/'
HASHES = {'segmentation.onnx': '220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079',
          'embedding.onnx': '1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b'}


def available():
    return all(p.is_file() and p.stat().st_size > 1024 for p in (SEGMENTATION, EMBEDDING))


def install(progress=print, cancel=None):
    cancel = cancel or threading.Event()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloads = [
        ('speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2',
         SEGMENTATION, 'sherpa-onnx-pyannote-segmentation-3-0/model.onnx'),
        ('speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx',
         EMBEDDING, None),
    ]
    for url, target, member in downloads:
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == HASHES[target.name]:
            continue
        part = target.with_suffix('.download')
        staged = target.with_suffix('.tmp')
        progress('正在下载说话人模型：' + target.name)
        try:
            with httpx.stream('GET', RELEASES + url, follow_redirects=True,
                              timeout=httpx.Timeout(60, connect=20)) as response:
                response.raise_for_status()
                with part.open('wb') as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        if cancel.is_set():
                            raise RuntimeError('已取消模型下载。')
                        output.write(chunk)
            if member:
                # Read exactly one known file, never extract arbitrary tar paths.
                with tarfile.open(part, 'r:bz2') as archive:
                    entry = archive.getmember(member)
                    if not entry.isfile() or entry.size > 100_000_000:
                        raise ValueError('模型包内容无效。')
                    with archive.extractfile(entry) as source, staged.open('wb') as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
            else:
                os.replace(part, staged)
            if hashlib.sha256(staged.read_bytes()).hexdigest() != HASHES[target.name]:
                raise ValueError('下载模型的校验值不匹配，请重试。')
            os.replace(staged, target)
        finally:
            part.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)
    progress('说话人模型已就绪（本地 CPU 并行分析）。')


if __name__ == '__main__':
    install()
    for path in (SEGMENTATION, EMBEDDING):
        print(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
