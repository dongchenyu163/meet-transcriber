from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from local_asr import timed_tokens,SenseVoiceTranscriber,available
from transcribe import Transcriber


class LocalASRTests(unittest.TestCase):
    def test_cuda_uses_fp32_provider_and_warms_up_before_ready(self):
        calls=[]
        recognizer=Mock()
        recognizer.decode_stream.side_effect=lambda _:calls.append('warmup')
        factory=Mock(side_effect=lambda **kwargs:(calls.append('create') or recognizer))
        with patch('local_asr.available',return_value=True), \
             patch('gpu_runtime.prepare_sensevoice_cuda',side_effect=lambda:calls.append('cuda')), \
             patch.dict('sys.modules',{'sherpa_onnx':SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_sense_voice=factory))}):
            SenseVoiceTranscriber(device='cuda',variant='small-fp32')
        self.assertEqual(calls,['cuda','create','warmup'])
        self.assertEqual(factory.call_args.kwargs['provider'],'cuda')
        self.assertTrue(factory.call_args.kwargs['model'].endswith('model.onnx'))
        recognizer.create_stream.return_value.accept_waveform.assert_called_once()

    def test_cpu_int8_never_initializes_cuda(self):
        factory=Mock()
        with patch('local_asr.available',return_value=True),patch('gpu_runtime.prepare_sensevoice_cuda') as cuda, \
             patch.dict('sys.modules',{'sherpa_onnx':SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_sense_voice=factory))}):
            SenseVoiceTranscriber()
        cuda.assert_not_called()
        self.assertEqual(factory.call_args.kwargs['provider'],'cpu')
        self.assertTrue(factory.call_args.kwargs['model'].endswith('model.int8.onnx'))

    def test_unsupported_cuda_int8_is_rejected_without_cpu_fallback(self):
        with self.assertRaisesRegex(ValueError,'FP32'):
            SenseVoiceTranscriber(device='cuda',variant='small-int8')

    def test_transcriber_passes_sensevoice_variant_and_device(self):
        with patch('local_asr.SenseVoiceTranscriber') as model:
            Transcriber(backend='sensevoice',device='cuda',sensevoice_variant='small-fp32').load()
        model.assert_called_once_with(None,4,'cuda','small-fp32')

    def test_unavailable_large_is_not_an_accepted_variant(self):
        with self.assertRaisesRegex(ValueError,'未知'):
            available('large')

    def test_sensevoice_timestamps_keep_cjk_and_english_spaces(self):
        result=SimpleNamespace(tokens=['你好','▁hello','日本語'],timestamps=[.1,1.,2.],text='你好 hello日本語')
        words=timed_tokens(result,3)
        self.assertEqual(''.join(w[2] for w in words),result.text)
        self.assertTrue(all(0<=s<e<=3 for s,e,_ in words))

    def test_missing_token_timestamps_fail_instead_of_inventing_times(self):
        with self.assertRaisesRegex(RuntimeError,'不完整'):
            timed_tokens(SimpleNamespace(tokens=['text'],timestamps=[],text='text'),2)

    def test_mixed_utterances_are_decoded_separately_with_original_offsets(self):
        model=object.__new__(SenseVoiceTranscriber)
        model.recognizer=Mock()
        streams=[SimpleNamespace(accept_waveform=Mock(),result=SimpleNamespace(tokens=[text],timestamps=[.1],text=text))
                 for text in ('中文','日本語','English')]
        model.recognizer.create_stream.side_effect=streams
        with patch('faster_whisper.vad.get_speech_timestamps',return_value=[
            {'start':0,'end':16000},{'start':32000,'end':48000},{'start':64000,'end':80000}]):
            words=model.transcribe(np.zeros(80000,np.float32))
        self.assertEqual([round(w[0],1) for w in words],[.1,2.1,4.1])
        self.assertEqual(''.join(w[2] for w in words),'中文日本語English')
        self.assertEqual(model.recognizer.decode_stream.call_count,3)

    def test_whisper_mixed_mode_redetects_each_utterance_and_restores_times(self):
        model=Transcriber(multilingual=True)
        model.model=Mock()
        model.model.transcribe.side_effect=[([SimpleNamespace(
            words=[SimpleNamespace(start=.1,end=.5,word=text)])],None) for text in ('中文','日本語','English')]
        with patch('local_asr.speech_regions',return_value=[{'start':i*16000,'end':(i+1)*16000} for i in range(3)]):
            words=model.transcribe(np.zeros(48000,np.float32))
        options=model.model.transcribe.call_args.kwargs
        self.assertEqual(model.model.transcribe.call_count,3)
        self.assertEqual([w[0] for w in words],[.1,1.1,2.1])
        self.assertEqual(''.join(w[2] for w in words),'中文日本語English')
        self.assertFalse(options['vad_filter'])
        self.assertIsNone(options['language'])


if __name__=='__main__':unittest.main()
