"""Capture/playback hardware checks using the same recording path as meetings."""
from datetime import datetime
from pathlib import Path
import time
import uuid
import numpy as np
from audio import AudioStore, AudioCapture, RATE
from settings import ROOT


def test_signal(rate=RATE):
    result=np.zeros(round(rate*3.2),np.float32)
    for start,frequency in ((.2,440), (1.2,660), (2.2,880)):
        length=round(rate*.6)
        envelope=np.ones(length)
        edge=round(rate*.02)
        envelope[:edge]=np.linspace(0,1,edge)
        envelope[-edge:]=np.linspace(1,0,edge)
        offset=round(start*rate)
        result[offset:offset+length]=.08*envelope*np.sin(2*np.pi*frequency*np.arange(length)/rate)
    return result


def play_signal(device_id,cancel):
    import soundcard as sc
    import sounddevice as sd
    speaker=sc.get_speaker(device_id) if device_id else sc.default_speaker()
    hosts=sd.query_hostapis()
    devices=[(i,d) for i,d in enumerate(sd.query_devices()) if d['name']==speaker.name
             and hosts[d['hostapi']]['name']=='Windows WASAPI' and d['max_output_channels']>0]
    if len(devices)!=1:raise RuntimeError('无法唯一匹配测试扬声器，请刷新设备并重新选择。')
    index,info=devices[0]
    rate=int(info['default_samplerate'])
    channels=min(2,info['max_output_channels'])
    samples=test_signal(rate)
    with sd.OutputStream(device=index,samplerate=rate,channels=channels,dtype='float32') as stream:
        for start in range(0,len(samples),rate//10):
            if cancel.is_set():return False
            block=np.repeat(samples[start:start+rate//10,None],channels,axis=1)
            if stream.write(block):raise RuntimeError('测试音播放发生缓冲欠载，请重试。')
    return True


def signal_correlation(samples):
    from scipy.signal import fftconvolve
    reference=test_signal()
    if len(samples)<len(reference):return 0.0
    cross=fftconvolve(samples,reference[::-1],mode='valid')
    energy=fftconvolve(samples.astype(np.float64)**2,np.ones(len(reference)),mode='valid')
    score=np.abs(cross)/np.sqrt(np.maximum(energy,1e-12)*float(np.sum(reference**2)))
    return min(1.,float(np.max(score)))


class HardwareTest:
    def __init__(self,loopback,device_id,cancel,emit,output_dir=None):
        self.loopback,self.device_id,self.cancel,self.emit=loopback,device_id,cancel,emit
        self.output_dir=Path(output_dir or ROOT/'test-output'/'hardware')

    def run(self):
        self.output_dir.mkdir(parents=True,exist_ok=True)
        source='system' if self.loopback else 'mic'
        path=self.output_dir/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:6]+'_'+source+'.wav')
        store=AudioStore(path)
        capture=AudioCapture(store,self.device_id,self.loopback)
        played=False
        capture.start()
        try:
            began=time.monotonic()
            while not capture.ready.wait(.05):
                error=capture.poll_error()
                if error:raise RuntimeError(error)
                if self.cancel.is_set():return None
                if time.monotonic()-began>15:raise RuntimeError('打开测试设备超时。')
            self.emit('status','系统测试：正在录制，稍后播放三个测试音…' if self.loopback else
                      '麦克风测试中，请说几句话，然后点击“停止测试并回放”。')
            began=time.monotonic()
            if self.loopback:
                if not self.cancel.wait(.5):
                    played=play_signal(self.device_id,self.cancel)
                    self.cancel.wait(.8)
            else:
                while not self.cancel.wait(.1) and time.monotonic()-began<120:
                    error=capture.poll_error()
                    if error:raise RuntimeError(error)
                    count,_,level=store.snapshot()
                    self.emit('hardware_progress',{'seconds':count/RATE,'elapsed':time.monotonic()-began,'level':level})
        finally:
            capture.stop()
            capture.join()
        if capture.error:raise RuntimeError(capture.error)
        count,epoch,_=store.snapshot()
        if count<RATE*.25:raise RuntimeError('录音过短，请至少录制一秒再停止。')
        samples,_=store.read(0,count)
        elapsed=time.time()-epoch
        duration=count/RATE
        rms=float(np.sqrt(np.mean(samples.astype(float)**2)))
        correlation=signal_correlation(samples) if self.loopback and played else None
        status=f'已录 {duration:.2f} 秒 / 经过 {elapsed:.2f} 秒；'
        if duration<max(0,elapsed-.8)*.9:status+='时长异常，请检查设备。'
        elif self.loopback and played:
            status+=f'测试音匹配度 {correlation:.0%}；'+('检测到测试音。' if correlation>=.45 else '未可靠检测到测试音，请检查输出设备和音量。')
        else:status+='检测到声音。' if rms>.0001 else '录音接近静音，请检查麦克风和音量。'
        return {'path':str(path),'seconds':duration,'elapsed':elapsed,'rms':rms,'correlation':correlation,'message':status}
