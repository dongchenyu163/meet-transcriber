"""Compare blocking and callback WASAPI timing; prints counts, saves no audio."""
import time
import sounddevice as sd
import soundcard as sc


if __name__=='__main__':
    name=sc.default_microphone().name
    hosts=sd.query_hostapis()
    matches=[(i,d) for i,d in enumerate(sd.query_devices()) if d['name']==name and hosts[d['hostapi']]['name']=='Windows WASAPI']
    for index,info in matches:
        print(name,info,flush=True)
        for channels in (1,2):
            for mode in ('callback','blocking'):
                counts=[]
                flags=[]
                def callback(data,frames,times,status):
                    counts.append((frames,times.inputBufferAdcTime))
                    if status:flags.append(str(status))
                try:
                    stream=sd.InputStream(device=index,samplerate=int(info['default_samplerate']),channels=channels,
                        dtype='float32',latency='high',callback=callback if mode=='callback' else None)
                    with stream:
                        start=time.perf_counter()
                        if mode=='callback':time.sleep(4)
                        else:
                            for _ in range(8):
                                data,overflow=stream.read(int(stream.samplerate)//4)
                                counts.append((len(data),0))
                                if overflow:flags.append('overflow')
                        wall=time.perf_counter()-start
                    print(channels,mode,'wall',round(wall,3),'audio',sum(n for n,_ in counts)/stream.samplerate,
                          'blocks',len(counts),'flags',flags[:3],flush=True)
                except Exception as exc:print(channels,mode,type(exc).__name__,str(exc),flush=True)
