import tempfile
from pathlib import Path
import queue
import threading
import unittest
from unittest.mock import patch
import numpy as np
from speaker_clustering import MeetingClustering
from speakers import SpeakerEngine, _speaker_worker
from audio import AudioStore, RATE
from storage import TranscriptWriter, recover_exports
from transcribe import TranscriptLine


def observation(vector, index):
    return {'start':1000+index*2, 'end':1002+index*2, 'vector':np.asarray(vector,np.float32)}


class MeetingClusteringTests(unittest.TestCase):
    def test_known_count_clusters_all_voices_instead_of_rejecting_after_slots_fill(self):
        basis=np.eye(4)
        model=MeetingClustering(3)
        model.update([observation(basis[i],i) for i in range(3)],final=True)
        # The same speaker with a large channel shift, below the old .6 cutoff.
        altered=basis[0]+2*basis[3]
        model.update([observation(altered,3)],final=True)
        self.assertEqual(model.previous[0]['speaker'],model.previous[3]['speaker'])
        self.assertEqual(len({t['speaker'] for t in model.previous.values()}),3)
        self.assertTrue(all(t['speaker'].startswith('SPEAKER_') for t in model.previous.values()))

    def test_late_speakers_recluster_earlier_fragments_and_keep_returning_voice_ids(self):
        basis=np.eye(4)
        model=MeetingClustering(3)
        # Early channel variation is initially mistaken for two distinct voices.
        model.update([observation(basis[0],0),observation(basis[0]+2*basis[3],1)])
        self.assertNotEqual(model.previous[0]['speaker'],model.previous[1]['speaker'])
        revised=model.update([observation(basis[1],2),observation(basis[2],3)],final=True)
        self.assertEqual(model.previous[0]['speaker'],model.previous[1]['speaker'])
        self.assertTrue(any(turn['id'] in ('voice-0','voice-1') for turn in revised))
        before={i:model.previous[i]['speaker'] for i in (0,2,3)}
        model.update([observation(basis[2],4),observation(basis[0],5),observation(basis[1],6)],final=True)
        self.assertEqual([model.previous[i]['speaker'] for i in (4,5,6)],
                         [before[3],before[0],before[2]])

    def test_live_window_with_one_voice_does_not_force_every_participant_to_appear(self):
        model=MeetingClustering(3)
        model.update([observation([1,i*.01,0],i) for i in range(5)])
        self.assertEqual(len(model.centers),1)

    def test_names_follow_global_groups_and_all_fragments_are_revised(self):
        model=MeetingClustering(2,{'Alice':[1,0,0], 'Bob':[0,1,0]})
        model.update([observation([1,0,0],0),observation([0,1,0],1)],final=True)
        model.update([observation([0,.9,.1],2),observation([.9,0,.1],3)],final=True)
        self.assertEqual([model.previous[i]['speaker_name'] for i in range(4)],['Alice','Bob','Bob','Alice'])

    def test_no_voice_and_overlap_remain_distinct_from_an_identity(self):
        model=MeetingClustering(2)
        items=[{'start':0,'end':1,'vector':None}, {'start':1,'end':2,'vector':None,'overlap':True}]
        result=model.update(items,final=True)
        self.assertEqual([t['speaker'] for t in result],['UNKNOWN','OVERLAP'])
        self.assertEqual(model.centers,{})

    def test_final_pass_supplies_requested_count_to_native_clustering(self):
        model=MeetingClustering(3)
        with patch.object(model,'_cluster',wraps=model._cluster) as cluster:
            model.update([observation(np.eye(4)[i%3],i) for i in range(12)],final=True)
        self.assertEqual(cluster.call_args.args[1],3)

    def test_long_meeting_bounds_pairwise_fit_but_assigns_every_observation(self):
        model=MeetingClustering(2,fit_budget=10)
        with patch.object(model,'_cluster',wraps=model._cluster) as cluster:
            model.update([observation(np.eye(2)[i%2],i) for i in range(101)],final=True)
        self.assertEqual(len(cluster.call_args.args[0]),10)
        self.assertEqual(len(model.previous),101)
        self.assertTrue(all(t['speaker'].startswith('SPEAKER_') for t in model.previous.values()))

    def test_separate_turn_features_can_undo_a_local_window_merge(self):
        engine=object.__new__(SpeakerEngine)
        # One locally merged turn spans two people. Independent features retain
        # that distinction for the later whole-meeting clustering pass.
        with patch.object(engine,'embed',side_effect=[np.array([1,0]),np.array([0,1])]):
            turns,vectors=engine._independent_turns(np.ones(RATE*8,np.float32),
                [{'start':0,'end':8,'cluster':0}],{0:np.array([1,1])})
        self.assertEqual(len(turns),2)
        np.testing.assert_array_equal(vectors[0],[1,0])
        np.testing.assert_array_equal(vectors[1],[0,1])

    def test_revised_speaker_events_replace_old_labels_after_journal_recovery(self):
        with tempfile.TemporaryDirectory(prefix='meet-recluster-') as folder:
            writer=TranscriptWriter(Path(folder),1000,{'diarize':True})
            writer.on_line(TranscriptLine(1000,1001,'first voice'))
            for speaker in ('SPEAKER_01','SPEAKER_02'):
                writer.append('speakers',{'upsert':True,'turns':[
                    {'id':'voice-0','start':1000,'end':1001,'speaker':speaker,'speaker_name':None}]})
            writer.finalize(1001)
            data=recover_exports(folder)
            self.assertEqual(len(data['speaker_turns']),1)
            self.assertEqual(data['lines'][0]['speaker'],'SPEAKER_02')

    def test_real_worker_path_uses_global_count_and_covers_final_tail(self):
        class Engine:
            def analyze(self,samples,independent=False):
                assert independent
                return [{'start':0,'end':len(samples)/RATE,'cluster':0}],{0:np.array([1,0])}
        with tempfile.TemporaryDirectory(prefix='meet-global-worker-') as folder:
            store=AudioStore(Path(folder)/'system.wav',started_at=1000)
            store.push(np.full(RATE*73,.1,np.float32))
            output=queue.Queue()
            ready,done,cancel=(threading.Event() for _ in range(3))
            done.set()
            with patch('speakers.SpeakerEngine',Engine),patch('speakers.load_profiles',return_value={}):
                _speaker_worker(store,output,ready,done,cancel,1)
            events=list(output.queue)
            self.assertFalse(any('error' in kind for kind,_ in events),events)
            snapshots={turn['id']:turn for kind,value in events if kind=='speakers' for turn in value['turns']}
            self.assertTrue(all(t['speaker']=='SPEAKER_01' for t in snapshots.values()))
            self.assertEqual(min(t['start'] for t in snapshots.values()),1000)
            self.assertEqual(max(t['end'] for t in snapshots.values()),1073)
            self.assertEqual(events[-1][0],'speaker_done')


if __name__=='__main__':unittest.main()
