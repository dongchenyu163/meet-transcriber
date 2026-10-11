import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtGui import QColor, QPalette
from gui import MainWindow, TaskThread
from hotkeys import GlobalHotkeys
from settings import ROOT
from storage import TranscriptWriter
from transcribe import TranscriptLine


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(load_inventory=False)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_initial_controls_and_password_field(self):
        self.assertTrue(self.window.start.isEnabled())
        self.assertFalse(self.window.stop.isEnabled())
        self.assertFalse(self.window.summarize.isEnabled())
        self.assertTrue(self.window.mic_enabled.isChecked())
        self.assertEqual(self.window.api_key.echoMode().name, 'Password')
        self.assertEqual(self.window.notion_key.echoMode().name,'Password')
        self.assertTrue(self.window.diarize.isChecked())
        self.assertEqual(self.window.config().speaker_count(),2)
        self.assertTrue(self.window.config().multilingual)
        self.assertFalse(self.window.hardware_stop.isEnabled())

    def test_settings_sections_fold_without_changing_values(self):
        section=self.window.settings_tabs.sections[0]
        self.window.model.setCurrentText('large-v3')
        section.set_expanded(False)
        self.assertTrue(section.body.isHidden())
        section.set_expanded(True)
        self.assertFalse(section.body.isHidden())
        self.assertEqual(self.window.model.currentText(),'large-v3')

    def test_hardware_stop_remains_available_during_test(self):
        self.window._set_busy(True,'hardware')
        self.assertTrue(self.window.hardware_stop.isEnabled())
        self.assertFalse(self.window.hardware_start.isEnabled())
        self.assertFalse(self.window.start.isEnabled())
        self.window.stop_hardware_test()
        self.assertTrue(self.window.cancel_task.is_set())
        self.window._set_busy(False)

    def test_sensevoice_gpu_selection_uses_fp32_and_keeps_whisper_model(self):
        self.window.model.setCurrentText('large-v3')
        self.window.asr_backend.setCurrentIndex(1)
        self.assertEqual(self.window.config().asr_backend,'sensevoice')
        self.assertEqual(self.window.config().device,'cpu')
        self.assertTrue(self.window.compute.isEnabled())
        self.window.compute.setCurrentIndex(1)
        config=self.window.config()
        self.assertEqual(config.device,'cuda')
        self.assertEqual(config.sensevoice_variant,'small-fp32')
        self.assertEqual(config.compute_type,'float32')
        self.window.asr_backend.setCurrentIndex(0)
        self.assertEqual(self.window.config().model_size,'large-v3')
        self.assertEqual(self.window.config().compute_type,'float16')

    def test_sensevoice_int8_switches_back_to_cpu(self):
        self.window.asr_backend.setCurrentIndex(1)
        self.window.compute.setCurrentIndex(1)
        self.window.sensevoice_variant.setCurrentIndex(0)
        self.assertEqual(self.window.config().device,'cpu')
        self.assertEqual(self.window.config().compute_type,'int8')

    def test_finished_hardware_test_automatically_replays_and_releases_controls(self):
        result={'path':str(ROOT/'test-output'/'test.wav'),'message':'已录 5 秒'}
        with patch('gui.HardwareTest') as hardware, patch.object(self.window,'replay_hardware') as replay:
            hardware.return_value.run.return_value=result
            self.window.start_hardware_test()
            for _ in range(100):
                self.app.processEvents()
                if not self.window.workers:break
                QTest.qWait(10)
            replay.assert_called_once()
        self.assertEqual(self.window.hardware_file,result['path'])
        self.assertIn('已录 5 秒',self.window.hardware_result.text())
        self.assertIsNone(self.window.active)
        self.assertTrue(self.window.hardware_replay.isEnabled())
        self.assertTrue(self.window.start.isEnabled())

    def test_light_theme_overrides_dark_system_palette(self):
        dark = QPalette()
        dark.setColor(QPalette.Window, QColor('#202020'))
        dark.setColor(QPalette.Base, QColor('#202020'))
        self.app.setPalette(dark)
        other = MainWindow(load_inventory=False)
        try:
            other.show()
            self.app.processEvents()
            self.assertEqual(self.app.style().objectName().lower(), 'fusion')
            self.assertGreater(other.palette().color(QPalette.Window).lightness(), 220)
            self.assertGreater(other.transcript.palette().color(QPalette.Base).lightness(), 220)
            other._set_busy(True, 'record')
            self.assertGreater(other.settings_tabs.palette().color(QPalette.Disabled, QPalette.Window).lightness(), 220)
        finally:
            other.close()

    def test_missing_api_key_blocks_network_and_points_to_settings(self):
        self.window.current_dir = ROOT
        self.window.api_key.clear()
        self.window.generate_minutes()
        self.assertIsNone(self.window.active)
        self.assertIn('API Key', self.window.status.text())
        self.assertEqual(self.window.settings_tabs.currentIndex(), 1)

    def test_gpu_selection_updates_precision(self):
        self.window.compute.setCurrentIndex(1)
        self.assertEqual(self.window.config().compute_type, 'float16')
        self.window.compute.setCurrentIndex(0)
        self.assertEqual(self.window.config().compute_type, 'int8')

    def test_worker_result_runs_without_blocking_gui(self):
        results = []
        worker = self.window.launch(lambda emit: 'done', results.append)
        for _ in range(100):
            self.app.processEvents()
            if not self.window.workers:
                break
            QTest.qWait(10)
        self.assertEqual(results, ['done'])
        self.assertEqual(self.window.workers, [])

    def test_history_load_shows_transcript(self):
        with tempfile.TemporaryDirectory(prefix='meet-gui-test-') as folder:
            root = Path(folder).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            writer = TranscriptWriter(root, 1000)
            writer.on_line(TranscriptLine(1001, 1002, 'test meeting'))
            writer.finalize(1003)
            self.window.load_session(root)
            self.assertIn('test meeting', self.window.transcript.toPlainText())
            self.assertTrue(self.window.summarize.isEnabled())

    def test_hotkey_partial_registration_is_cleaned_up(self):
        keys = GlobalHotkeys(self.app, lambda: None, lambda: None)
        with patch('hotkeys.ctypes.windll.user32.RegisterHotKey', side_effect=[1, 0]), \
             patch('hotkeys.ctypes.windll.user32.UnregisterHotKey') as unregister:
            warnings = keys.install()
            self.assertEqual(len(warnings), 1)
            self.assertIn('F8', warnings[0])
            self.assertEqual(keys.registered, [0x4D01])
            keys.close()
            unregister.assert_called_once_with(None, 0x4D01)

    def test_render_application(self):
        output = ROOT / 'test-output'
        output.mkdir(exist_ok=True)
        self.assertTrue(self.window.grab().save(str(output / 'interface.png')))
        self.assertGreater(self.window.transcript.width(), 400)
        for index,name in [(2,'notion'),(3,'speakers'),(4,'hardware')]:
            self.window.settings_tabs.setCurrentIndex(index)
            self.app.processEvents()
            self.assertTrue(self.window.grab().save(str(output/f'interface-{name}.png')))
        self.window.settings_tabs.setCurrentIndex(0)
        self.window.asr_backend.setCurrentIndex(1)
        self.window.compute.setCurrentIndex(1)
        self.app.processEvents()
        self.window.settings_tabs.ensureWidgetVisible(self.window.sensevoice_variant)
        self.app.processEvents()
        self.assertTrue(self.window.grab().save(str(output/'interface-sensevoice-gpu.png')))

    def test_missing_notion_token_blocks_sync_before_network(self):
        self.window.current_dir = ROOT
        self.window.notion_key.clear()
        self.window.upload_notion()
        self.assertIsNone(self.window.active)
        self.assertEqual(self.window.settings_tabs.currentIndex(),2)
        self.assertIn('Token',self.window.status.text())

    def test_auto_upload_does_not_follow_a_changed_meeting(self):
        self.window.current_dir = ROOT/'different'
        with patch.object(self.window,'upload_notion') as upload:
            self.window._auto_upload(ROOT/'original')
        upload.assert_not_called()


if __name__ == '__main__':
    unittest.main()
