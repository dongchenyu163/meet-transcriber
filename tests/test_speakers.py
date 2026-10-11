import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from attribution import attribute, speaker_label
from audio import AudioStore, RATE
from speakers import SpeakerTracker, _speaker_worker
from storage import TranscriptWriter, recover_exports, rename_speaker
from transcribe import TranscriptLine
from summary import transcript_text


class SpeakerTests(unittest.TestCase):
    def test_participant_count_accounts_for_separate_microphone(self):
        from session import SessionConfig
        self.assertEqual(SessionConfig(participants=3).speaker_count(),2)
        self.assertEqual(SessionConfig(participants=3,record_mic=False).speaker_count(),3)
        self.assertEqual(SessionConfig(participants=3,mic_is_participant=False).speaker_count(),3)
        self.assertEqual(SessionConfig(participants=0).speaker_count(),0)
        with self.assertRaises(ValueError):SessionConfig(participants=1).speaker_count()

    def test_word_boundaries_split_speakers_without_losing_text(self):
        line = {'start':0,'end':4,'source':'system','text':'你好再见', 'words':[
            {'start':0,'end':2,'text':'你好'}, {'start':2,'end':4,'text':'再见'}]}
        turns = [{'start':0,'end':2,'speaker':'SPEAKER_01'}, {'start':2,'end':4,'speaker':'SPEAKER_02'}]
        lines = attribute([line],turns,{'SPEAKER_02':'李四'},True)
        self.assertEqual(''.join(x['text'] for x in lines),line['text'])
        self.assertEqual([x['speaker'] for x in lines],['SPEAKER_01','SPEAKER_02'])
        self.assertEqual(speaker_label(lines[1]),'李四')

    def test_overlap_is_not_falsely_assigned_to_one_person(self):
        line = {'start':0,'end':2,'text':'hello','source':'system'}
        turns = [{'start':0,'end':2,'speaker':'SPEAKER_01'}, {'start':.1,'end':1.9,'speaker':'SPEAKER_02'}]
        self.assertEqual(attribute([line],turns,enabled=True)[0]['speaker'],'OVERLAP')

    def test_unknown_and_mic_preserve_original_text(self):
        lines = [{'start':0,'end':2,'text':'system','source':'system'},
                 {'start':0,'end':2,'text':'mic','source':'mic'}]
        result = attribute(lines,[],enabled=True)
        self.assertEqual(result[0]['speaker'],'UNKNOWN')
        self.assertEqual(result[1],lines[1])

    def test_tracker_keeps_ids_when_window_cluster_numbers_change(self):
        tracker = SpeakerTracker({'Alice':[1,0,0]})
        first = tracker.assign({0:np.array([1,0,0]),1:np.array([0,1,0])})
        second = tracker.assign({8:np.array([0,1,0]),3:np.array([1,0,0])})
        self.assertEqual(first[0],second[3])
        self.assertEqual(first[1],second[8])
        self.assertEqual(first[0]['speaker_name'],'Alice')
        self.assertIsNone(first[1]['speaker_name'])

    def test_overlapping_clusters_cannot_merge_and_ambiguous_name_is_unknown(self):
        tracker = SpeakerTracker({'Alice':[1,0], 'Bob':[1,.01]})
        labels = tracker.assign({0:np.array([1,0]),1:np.array([1,.01])},[(0,1)])
        self.assertNotEqual(labels[0]['speaker'],labels[1]['speaker'])
        self.assertIsNone(labels[0]['speaker_name'])

    def test_journal_recovery_and_summary_retain_manual_names(self):
        with tempfile.TemporaryDirectory(prefix='meet-speaker-test-') as tmp:
            root = Path(tmp)
            writer = TranscriptWriter(root,1000,{'diarize':True})
            writer.on_line(TranscriptLine(1000,1002,'交付报告'))
            writer.append('speakers',{'start':1000,'end':1002,'turns':[
                {'start':1000,'end':1002,'speaker':'SPEAKER_01','speaker_name':None}]})
            writer.finalize(1003)
            rename_speaker(root,'SPEAKER_01','张三')
            data = recover_exports(root)
            self.assertIn('张三',transcript_text(data))
            self.assertIn('张三',(root/'transcript.txt').read_text(encoding='utf-8'))

    def test_worker_visits_all_windows_and_short_tail(self):
        class Engine:
            def analyze(self,samples):
                return [{'start':0,'end':len(samples)/RATE,'cluster':0}],{0:np.array([1,0])}
        with tempfile.TemporaryDirectory(prefix='meet-speaker-test-') as tmp:
            store = AudioStore(Path(tmp)/'system.wav',started_at=1000)
            store.push(np.full(RATE*73,.1,dtype=np.float32))
            output = queue.Queue()
            ready,done,cancel = (threading.Event() for _ in range(3))
            done.set()
            with patch('speakers.SpeakerEngine',Engine), patch('speakers.load_profiles',return_value={}):
                _speaker_worker(store,output,ready,done,cancel)
            events = list(output.queue)
            windows = [value for kind,value in events if kind=='speakers']
            self.assertEqual(windows[0]['start'],1000)
            self.assertEqual(windows[-1]['end'],1073)
            self.assertTrue(all(a['end']==b['start'] for a,b in zip(windows,windows[1:])))
            self.assertEqual(events[-1][0],'speaker_done')


if __name__ == '__main__': unittest.main()
