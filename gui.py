"""PySide6 desktop interface. All long-running operations use worker threads."""
from datetime import datetime
import json
from pathlib import Path
import threading
import time

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap, QFontDatabase
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QFormLayout, QLabel, QPushButton, QComboBox, QLineEdit,
    QCheckBox, QSpinBox, QTextEdit, QPlainTextEdit, QTabWidget, QSplitter,
    QFileDialog, QProgressBar, QScrollArea, QFrame)

from audio import list_devices
from hotkeys import GlobalHotkeys
from screenshot import list_windows
from session import MeetingSession, SessionConfig, retranscribe
from settings import ROOT, DeepSeekSettings, NotionSettings
from storage import recover_exports, SOURCE_LABELS, rename_speaker
from summary import DeepSeekSummarizer
from theme import apply_light_theme
from notion_sync import NotionSync
from speaker_models import available as speaker_models_available, install as install_speaker_models
from speakers import enroll, load_profiles
from widgets import SettingsSections, CollapsibleSection
from soundcheck import HardwareTest
from local_asr import available as sensevoice_available, install as install_sensevoice, MODELS as SENSEVOICE_MODELS


class TaskThread(QThread):
    event = Signal(str, object)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.result.emit(self.function(self.event.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, load_inventory=True):
        apply_light_theme(QApplication.instance())
        # The offscreen Qt backend on Windows starts with an empty font database.
        # Loading installed fonts also makes exported UI previews readable.
        if not QFontDatabase.families():
            for name in ('msyh.ttc', 'msyhbd.ttc', 'segoeui.ttf'):
                font_path = Path('C:/Windows/Fonts') / name
                if font_path.exists():
                    QFontDatabase.addApplicationFont(str(font_path))
        super().__init__()
        self.setWindowTitle('Meet Notes · 会议记录')
        self.resize(1180, 800)
        self.setMinimumSize(940, 660)
        self.workers = []
        self.active = None
        self.session = None
        self.current_dir = None
        self.cancel_task = threading.Event()
        self.hotkeys = None
        self.closing = False
        self.hardware_file = None
        self.player = None
        self._build_ui()
        self._load_api()
        if load_inventory:
            QTimer.singleShot(0, self.refresh_devices)

    def _build_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        layout.setContentsMargins(24, 20, 24, 18)
        heading = QLabel('Meet Notes')
        heading.setObjectName('heading')
        layout.addWidget(heading)
        layout.addWidget(QLabel('本地转录  /  重点截图  /  DeepSeek 会议纪要'))
        self.status = QLabel('准备就绪 · 选择音频设备后开始会议')
        self.status.setObjectName('status')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split, 1)
        self.settings_tabs = SettingsSections()
        self.settings_tabs.setMinimumWidth(390)
        split.addWidget(self.settings_tabs)
        record_page = QWidget()
        record_layout = QVBoxLayout(record_page)
        def record_section(title, expanded=True):
            page=QWidget()
            section=CollapsibleSection(title,page,expanded)
            record_layout.addWidget(section)
            return QFormLayout(page)
        form = record_section('音频设备')
        form.setVerticalSpacing(12)
        self.system_enabled = QCheckBox('录制会议播放声')
        self.system_enabled.setChecked(True)
        self.system_device = QComboBox()
        self.system_device.addItem('系统默认扬声器', None)
        self.mic_enabled = QCheckBox('录制自己的麦克风')
        self.mic_enabled.setChecked(True)
        self.mic_device = QComboBox()
        self.mic_device.addItem('系统默认麦克风', None)
        form.addRow(self.system_enabled)
        form.addRow(self.system_device)
        form.addRow(self.mic_enabled)
        form.addRow(self.mic_device)
        note = QLabel('请让会议声音输出到所选扬声器。建议佩戴耳机，避免麦克风重复录入外放声音。')
        note.setWordWrap(True)
        note.setObjectName('muted')
        form.addRow(note)
        self.refresh = QPushButton('刷新设备与窗口')
        self.refresh.clicked.connect(self.refresh_devices)
        form.addRow(self.refresh)
        form = record_section('转录模型')
        self.asr_backend = QComboBox()
        self.asr_backend.addItem('Whisper · 本地多语言', 'whisper')
        self.asr_backend.addItem('SenseVoiceSmall · 本地中日英', 'sensevoice')
        form.addRow('识别网络',self.asr_backend)
        self.model = QComboBox()
        self.model.addItems(['small', 'medium', 'large-v3', 'tiny', 'base'])
        self.model.setEditable(True)
        self.whisper_model_label = QLabel('Whisper 模型')
        form.addRow(self.whisper_model_label, self.model)
        self.sensevoice_variant = QComboBox()
        for key, spec in SENSEVOICE_MODELS.items():
            self.sensevoice_variant.addItem(spec['label'], key)
        self.sensevoice_model_label = QLabel('SenseVoice 版本')
        form.addRow(self.sensevoice_model_label, self.sensevoice_variant)
        self.sensevoice_variant.currentIndexChanged.connect(self._sensevoice_variant_changed)
        self.language = QComboBox()
        for label, value in [('中日英混说（逐段检测）', 'mixed'), ('自动检测', None), ('中文', 'zh'), ('English', 'en'), ('日本語', 'ja')]:
            self.language.addItem(label, value)
        form.addRow('主要语言', self.language)
        self.compute = QComboBox()
        self.compute.addItem('CPU', 'cpu')
        self.compute.addItem('NVIDIA GPU（需 CUDA）', 'cuda')
        self.compute.currentIndexChanged.connect(self._compute_changed)
        form.addRow('推理设备', self.compute)
        self.precision = QComboBox()
        self.precision.addItems(['int8', 'float16', 'int8_float16', 'float32'])
        form.addRow('计算精度', self.precision)
        self.threads = QSpinBox()
        self.threads.setRange(1, 64)
        self.threads.setValue(4)
        form.addRow('CPU 线程', self.threads)
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText('人名、项目名、专业术语（可选）')
        form.addRow('识别提示', self.prompt)
        self.sensevoice_install=QPushButton('下载 / 检查所选 SenseVoice 模型')
        self.sensevoice_install.clicked.connect(self.install_sensevoice_model)
        form.addRow(self.sensevoice_install)
        self.backend_note=QLabel()
        self.backend_note.setWordWrap(True)
        self.backend_note.setObjectName('muted')
        form.addRow(self.backend_note)
        self.asr_backend.currentIndexChanged.connect(self._backend_changed)
        self._backend_changed()
        form = record_section('截图与保存',False)
        self.window = QComboBox()
        self.window.addItem('请选择会议窗口', None)
        form.addRow('截图窗口', self.window)
        self.full_screen = QCheckBox('整屏截图（包含其他应用）')
        self.full_screen.toggled.connect(lambda checked: self.window.setEnabled(not checked))
        form.addRow(self.full_screen)
        self.interval = QSpinBox()
        self.interval.setRange(0, 3600)
        self.interval.setSuffix(' 秒')
        self.interval.setSpecialValueText('关闭自动截图')
        form.addRow('自动截图间隔', self.interval)
        self.output = QLineEdit(str(ROOT / 'sessions'))
        browse = QPushButton('选择保存位置')
        browse.clicked.connect(self.choose_output)
        form.addRow('保存目录', self.output)
        form.addRow(browse)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(record_page)
        self.settings_tabs.addTab(scroll, '录制')

        api_page = QWidget()
        api = QFormLayout(api_page)
        api.setVerticalSpacing(14)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText('填写你的 DeepSeek API Key')
        self.api_url = QLineEdit()
        self.api_model = QLineEdit()
        self.api_timeout = QSpinBox()
        self.api_timeout.setRange(5, 600)
        self.api_timeout.setSuffix(' 秒')
        api.addRow('API Key', self.api_key)
        api.addRow('API 地址', self.api_url)
        api.addRow('模型名称', self.api_model)
        api.addRow('请求超时', self.api_timeout)
        save = QPushButton('保存 API 配置')
        save.clicked.connect(self.save_api)
        api.addRow(save)
        self.auto_summary = QCheckBox('会议结束后自动生成纪要')
        api.addRow(self.auto_summary)
        info = QLabel('生成纪要时，会把转录文字发送到配置的 DeepSeek API；音频和截图保留在本机。\n\nAPI Key 保存在本机 .env 文件中（明文），不会写入会议记录。系统环境变量的配置优先。')
        info.setWordWrap(True)
        info.setObjectName('muted')
        api.addRow(info)
        self.settings_tabs.addTab(api_page, 'DeepSeek')

        notion_page = QWidget()
        notion = QFormLayout(notion_page)
        notion.setVerticalSpacing(12)
        self.notion_key = QLineEdit()
        self.notion_key.setEchoMode(QLineEdit.Password)
        self.notion_key.setPlaceholderText('Internal connection Token')
        self.notion_parent = QLineEdit()
        self.notion_parent.setPlaceholderText('目标页面链接或页面 ID')
        self.notion_auto = QCheckBox('生成纪要后自动同步到 Notion')
        notion.addRow('Token', self.notion_key)
        notion.addRow('目标页面', self.notion_parent)
        notion.addRow(self.notion_auto)
        save_notion = QPushButton('保存 Notion 配置')
        save_notion.clicked.connect(self.save_notion)
        test_notion = QPushButton('测试连接（读取目标页面）')
        test_notion.clicked.connect(self.test_notion)
        self.sync_notion = QPushButton('同步当前会议纪要')
        self.sync_notion.clicked.connect(self.upload_notion)
        self.notion_link = QLabel('尚未同步')
        self.notion_link.setWordWrap(True)
        self.notion_link.setOpenExternalLinks(True)
        for item in (save_notion, test_notion, self.sync_notion, self.notion_link):
            notion.addRow(item)
        note = QLabel('每场会议在目标页面下创建一个子页面，再次同步会更新对应子页面。\n\n请为内部连接开启读取、插入和更新内容权限，并把目标页面共享给连接。Token 明文保存在本机 .env，系统环境变量优先。\n\n同步只上传生成的 Markdown 纪要；本地截图链接不会自动上传图片。')
        note.setWordWrap(True)
        note.setObjectName('muted')
        notion.addRow(note)
        self.settings_tabs.addTab(notion_page, 'Notion')

        speakers_page = QWidget()
        speakers_form = QFormLayout(speakers_page)
        self.diarize = QCheckBox('会议中同时区分说话人')
        self.diarize.setChecked(True)
        speakers_form.addRow(self.diarize)
        self.participants=QSpinBox()
        self.participants.setRange(0,100)
        self.participants.setSpecialValueText('自动估计人数')
        self.participants.setSuffix(' 人')
        self.participants.setValue(3)
        speakers_form.addRow('总开会人数（含自己）',self.participants)
        self.mic_is_participant=QCheckBox('自己的麦克风单独占一人')
        self.mic_is_participant.setChecked(True)
        speakers_form.addRow(self.mic_is_participant)
        limit_note=QLabel('例：总共 3 人且自己的麦克风单独占一人，则按 2 位远端发言人聚类。程序会结合整场声纹修正早先归属，结束时统一重分组。请按实际发言人数填写；0 为自动估计。')
        limit_note.setWordWrap(True)
        limit_note.setObjectName('muted')
        speakers_form.addRow(limit_note)
        self.speaker_progress = QLabel('CPU 并行分析 · 约 24 秒后开始显示暂定归属')
        self.speaker_progress.setWordWrap(True)
        speakers_form.addRow(self.speaker_progress)
        self.speaker_install = QPushButton('下载 / 检查说话人模型')
        self.speaker_install.clicked.connect(self.install_speakers)
        speakers_form.addRow(self.speaker_install)
        self.voice_name = QLineEdit()
        self.voice_name.setPlaceholderText('声音样本对应的姓名')
        speakers_form.addRow('姓名', self.voice_name)
        register = QPushButton('选择声音样本并登记')
        register.clicked.connect(self.register_voice)
        speakers_form.addRow(register)
        self.voice_list = QLabel()
        self.voice_list.setWordWrap(True)
        speakers_form.addRow(self.voice_list)
        self.speaker_select = QComboBox()
        self.speaker_rename = QLineEdit()
        self.speaker_rename.setPlaceholderText('手动确认的姓名')
        speakers_form.addRow('当前会议说话人', self.speaker_select)
        speakers_form.addRow('修改姓名', self.speaker_rename)
        rename = QPushButton('保存当前会议姓名')
        rename.clicked.connect(self.rename_current_speaker)
        speakers_form.addRow(rename)
        note = QLabel('Whisper 可使用 GPU，说话人分析使用 CPU 独立进程，两者同时运行，不额外占用显存。\n\n人数用于约束整场声音分组；实时归属可能随新发言修正。无法提取声纹的片段显示“未知”，重叠人声单独标注。姓名通过登记样本匹配，也可会后手动修改。\n\n登记样本应仅一人说话，建议每人 30～60 秒。只在本机保存声纹；同名登记会替换原声纹。')
        note.setWordWrap(True)
        note.setObjectName('muted')
        speakers_form.addRow(note)
        speaker_scroll = QScrollArea()
        speaker_scroll.setWidgetResizable(True)
        speaker_scroll.setWidget(speakers_page)
        self.settings_tabs.addTab(speaker_scroll, '说话人')

        self.hardware_page=QWidget()
        hardware=QFormLayout(self.hardware_page)
        self.hardware_source=QComboBox()
        self.hardware_source.addItem('麦克风（使用上方所选设备）',False)
        self.hardware_source.addItem('系统播放声（使用上方所选扬声器）',True)
        hardware.addRow(self.hardware_source)
        self.hardware_start=QPushButton('开始硬件测试')
        self.hardware_start.clicked.connect(self.start_hardware_test)
        self.hardware_stop=QPushButton('停止测试并回放')
        self.hardware_stop.clicked.connect(self.stop_hardware_test)
        self.hardware_replay=QPushButton('再次回放最近的测试录音')
        self.hardware_replay.clicked.connect(self.replay_hardware)
        self.hardware_stop_play=QPushButton('停止回放')
        self.hardware_stop_play.clicked.connect(self.stop_playback)
        self.hardware_level=QProgressBar()
        self.hardware_level.setRange(0,100)
        self.hardware_level.setValue(0)
        self.hardware_level.setTextVisible(False)
        self.hardware_level.setFixedHeight(8)
        self.hardware_result=QLabel('麦克风：说几句话后点击停止，将自动回放。\n系统声音：自动播放三个轻声测试音，几秒后停止录制并回放。测试音请保持正常系统音量。')
        self.hardware_result.setWordWrap(True)
        for widget in (self.hardware_start,self.hardware_stop,self.hardware_level,self.hardware_result,
                       self.hardware_replay,self.hardware_stop_play):hardware.addRow(widget)
        note=QLabel('录音保存在本机 test-output/hardware，可反复回放。系统测试录制整个所选输出设备，包含该设备上的其他应用声音，不是单独的会议窗口。测试最长 120 秒，不加载识别模型。')
        note.setWordWrap(True)
        note.setObjectName('muted')
        hardware.addRow(note)
        self.settings_tabs.addTab(self.hardware_page,'硬件测试与回放')

        right = QWidget()
        content = QVBoxLayout(right)
        meters = QHBoxLayout()
        self.levels, self.backlogs = {}, {}
        for source, label in [('system', '会议声音'), ('mic', '麦克风')]:
            box = QVBoxLayout()
            box.addWidget(QLabel(label))
            meter = QProgressBar()
            meter.setRange(0, 100)
            meter.setValue(0)
            meter.setTextVisible(False)
            meter.setFixedHeight(8)
            self.levels[source] = meter
            box.addWidget(meter)
            backlog = QLabel('等待录音')
            backlog.setObjectName('muted')
            self.backlogs[source] = backlog
            box.addWidget(backlog)
            meters.addLayout(box)
        content.addLayout(meters)
        self.tabs = QTabWidget()
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText('会议开始后，识别出的文字会显示在这里。\n音频始终保存，可在结束后重新转录。')
        self.minutes = QTextEdit()
        self.minutes.setReadOnly(True)
        self.minutes.setPlaceholderText('会议结束后，点击“生成纪要”。')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.shot_preview = QLabel('截图预览')
        self.shot_preview.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.transcript, '会议转录')
        self.tabs.addTab(self.minutes, '会议纪要')
        self.tabs.addTab(self.shot_preview, '重点截图')
        self.tabs.addTab(self.log, '运行信息')
        content.addWidget(self.tabs, 1)
        self.session_path = QLabel('尚未创建会议')
        self.session_path.setWordWrap(True)
        self.session_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content.addWidget(self.session_path)
        split.addWidget(right)
        split.setSizes([390, 740])
        buttons = QHBoxLayout()
        self.start = QPushButton('开始会议')
        self.start.setObjectName('primary')
        self.stop = QPushButton('结束并补完转录 · F8')
        self.shot = QPushButton('重点截图 · F9')
        self.save_only = QPushButton('仅保存录音')
        self.summarize = QPushButton('生成纪要')
        self.recover = QPushButton('重新转录')
        self.open_session = QPushButton('打开历史会议')
        self.open_folder = QPushButton('打开文件夹')
        for button in [self.start, self.stop, self.shot, self.save_only, self.summarize,
                       self.recover, self.open_session, self.open_folder]:
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.start.clicked.connect(self.start_meeting)
        self.stop.clicked.connect(self.stop_meeting)
        self.shot.clicked.connect(self.keyshot)
        self.save_only.clicked.connect(self.save_audio_only)
        self.summarize.clicked.connect(self.generate_minutes)
        self.recover.clicked.connect(self.recover_transcript)
        self.open_session.clicked.connect(self.choose_session)
        self.open_folder.clicked.connect(self.show_folder)
        self._set_busy(False)
        self.setStyleSheet('''
            QMainWindow { background: #f5f7fb; }
            QToolButton#sectionHeader { text-align: left; padding: 10px; border: 1px solid #d6dfec; border-radius: 5px; background: #edf3fb; font-weight: 600; }
            QWidget { color: #23324b; font-family: 'Microsoft YaHei UI'; font-size: 12px; }
            QLabel#heading { font-size: 29px; font-weight: 700; color: #173862; }
            QLabel#status { padding: 12px; background: #e7effa; border-radius: 8px; color: #214d83; }
            QLabel#muted { color: #6c7890; font-size: 11px; }
            QPushButton { padding: 9px 10px; background: white; border: 1px solid #d6dfec; border-radius: 6px; }
            QPushButton:hover { background: #edf3ff; }
            QPushButton:disabled { color: #9da8bb; background: #f0f2f6; }
            QPushButton#primary { background: #2763bd; color: white; border: 0; }
            QPushButton#primary:disabled { background: #9bb5d8; }
            QLineEdit, QComboBox, QSpinBox { padding: 6px; background: white; border: 1px solid #d6dfec; border-radius: 5px; }
            QTextEdit, QPlainTextEdit { border: 1px solid #dce3ee; background: white; padding: 10px; font-size: 14px; }
            QTabWidget::pane { border: 1px solid #dce3ee; background: white; }
            QTabBar::tab { padding: 10px 16px; background: #edf1f7; color: #53627c; }
            QTabBar::tab:selected { color: #2763bd; background: white; }
            QTabBar::tab:disabled { color: #778398; background: #edf1f7; }
            QScrollArea { border: 1px solid #dce3ee; background: #f5f7fb; }
            QScrollBar:vertical { background: #edf1f7; width: 12px; margin: 0; }
            QScrollBar::handle:vertical { background: #b5c2d5; min-height: 24px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: #edf1f7; }
            QSplitter::handle { background: #dce3ee; }
            QComboBox QAbstractItemView { background: white; color: #23324b; selection-background-color: #2763bd; selection-color: white; }
            QProgressBar { background: #dce4ef; border: 0; border-radius: 4px; }
            QProgressBar::chunk { background: #35a18a; border-radius: 4px; }
        ''')

    def _compute_changed(self):
        if hasattr(self, 'precision'):
            if self.asr_backend.currentData() == 'sensevoice':
                if self.compute.currentData() == 'cuda':
                    self.sensevoice_variant.setCurrentIndex(self.sensevoice_variant.findData('small-fp32'))
                self._sensevoice_variant_changed()
            else:
                self.precision.setCurrentText('float16' if self.compute.currentData() == 'cuda' else 'int8')

    def _sensevoice_variant_changed(self):
        if self.asr_backend.currentData() != 'sensevoice' or not hasattr(self, 'precision'):return
        is_int8 = self.sensevoice_variant.currentData() == 'small-int8'
        if is_int8 and self.compute.currentData() == 'cuda':
            self.compute.setCurrentIndex(self.compute.findData('cpu'))
        self.precision.setCurrentText('int8' if is_int8 else 'float32')

    def _backend_changed(self):
        local_sensevoice=self.asr_backend.currentData()=='sensevoice'
        for widget in (self.model,self.precision,self.prompt):widget.setEnabled(not local_sensevoice)
        for widget in (self.model, self.whisper_model_label):widget.setVisible(not local_sensevoice)
        for widget in (self.sensevoice_variant, self.sensevoice_model_label):widget.setVisible(local_sensevoice)
        self._compute_changed()
        self.sensevoice_install.setVisible(local_sensevoice)
        self.backend_note.setText('INT8 用于 CPU；FP32 可选 CPU / NVIDIA GPU。首次启用 GPU 请运行 setup-sensevoice-gpu.bat。FP32 仍是 Small；官方尚未公开 Large 权重。音频不上传。' if local_sensevoice else
            'Whisper 支持 GPU；混说时推荐 large-v3 与“中日英混说”，保留各语言原文。')

    def install_sensevoice_model(self):
        if self.active:return
        self.cancel_task.clear()
        variant = self.sensevoice_variant.currentData()
        self.launch(lambda emit: install_sensevoice(lambda text:emit('status',text),self.cancel_task,variant),mode='speakers')

    def stop_playback(self):
        if self.player:self.player.stop()

    def replay_hardware(self):
        if not self.hardware_file or self.closing:return
        from PySide6.QtMultimedia import QMediaPlayer,QAudioOutput
        if self.player is None:
            self.player=QMediaPlayer(self)
            self.audio_output=QAudioOutput(self)
            self.audio_output.setVolume(.7)
            self.player.setAudioOutput(self.audio_output)
            self.player.errorOccurred.connect(lambda error,message:self.handle_event('warning','回放失败：'+message))
        self.player.setSource(QUrl.fromLocalFile(str(Path(self.hardware_file).resolve())))
        self.player.play()

    def start_hardware_test(self):
        if self.active:return
        self.stop_playback()
        self.cancel_task.clear()
        loopback=self.hardware_source.currentData()
        device=self.system_device.currentData() if loopback else self.mic_device.currentData()
        self.hardware_result.setText('正在打开测试设备…')
        self.hardware_level.setValue(0)
        def done(result):
            if not result:return
            self.hardware_file=result['path']
            self.hardware_result.setText(result['message']+'\n'+result['path'])
            self.status.setText(result['message']+' 正在回放录音。')
            self.replay_hardware()
        self.launch(lambda emit:HardwareTest(loopback,device,self.cancel_task,emit).run(),done,'hardware')

    def stop_hardware_test(self):
        self.cancel_task.set()
        self.hardware_stop.setEnabled(False)
        self.hardware_result.setText('正在停止录制，随后回放…')

    def _load_api(self):
        settings = DeepSeekSettings.load()
        self.api_key.setText(settings.api_key)
        self.api_url.setText(settings.base_url)
        self.api_model.setText(settings.model)
        self.api_timeout.setValue(int(settings.timeout))
        notion = NotionSettings.load()
        self.notion_key.setText(notion.api_key)
        self.notion_parent.setText(notion.parent_page)
        self.notion_auto.setChecked(notion.auto_sync)
        self.refresh_voices()

    def notion_settings(self):
        return NotionSettings(self.notion_key.text().strip(), self.notion_parent.text().strip(),
                              self.notion_auto.isChecked())

    def save_notion(self):
        try:
            self.notion_settings().save()
            self.status.setText('Notion 配置已保存到本机 .env')
        except Exception as exc:
            self.handle_event('error', str(exc))

    def test_notion(self):
        if self.active:
            return
        try:
            settings = self.notion_settings()
            settings.validate()
        except Exception as exc:
            self.handle_event('error', str(exc))
            return
        self.cancel_task.clear()
        self.launch(lambda emit: NotionSync(settings, self.cancel_task).check_connection(),
            lambda _: self.status.setText('Notion 目标页面可读取；首次同步时会验证写入权限。'), 'notion')

    def upload_notion(self):
        if self.active or not self.current_dir:
            return
        try:
            settings = self.notion_settings()
            settings.validate()
            if not (self.current_dir / 'minutes.md').exists():
                raise ValueError('请先生成当前会议纪要。')
        except Exception as exc:
            self.handle_event('error', str(exc))
            self.settings_tabs.setCurrentIndex(2)
            return
        root = self.current_dir
        self.cancel_task.clear()
        def done(url):
            self.notion_link.setText(f'<a href="{url}">打开已同步的 Notion 会议纪要</a>')
            self.status.setText('会议纪要已同步到 Notion')
        self.launch(lambda emit: NotionSync(settings, self.cancel_task,
            lambda message: emit('status', message)).sync(root), done, 'notion')

    def refresh_voices(self):
        try:
            names = list(load_profiles())
            self.voice_list.setText('已登记：' + ('、'.join(names) if names else '无；仍可区分说话人编号'))
        except Exception as exc:
            self.voice_list.setText(str(exc))

    def install_speakers(self):
        if self.active:
            return
        self.cancel_task.clear()
        self.launch(lambda emit: install_speaker_models(lambda text: emit('status', text), self.cancel_task),
                    mode='speakers')

    def register_voice(self):
        if self.active:
            return
        name = self.voice_name.text().strip()
        if not name:
            self.handle_event('error', '请先填写声音样本对应的姓名。')
            return
        files, _ = QFileDialog.getOpenFileNames(self, '选择仅一人说话的录音（建议合计 30～60 秒）', '',
            '音频文件 (*.wav *.flac *.mp3 *.m4a *.ogg);;所有文件 (*)')
        if not files:
            return
        def done(message):
            self.status.setText(message)
            self.refresh_voices()
        self.cancel_task.clear()
        self.launch(lambda emit: enroll(name, files, lambda text: emit('status', text), self.cancel_task), done, 'speakers')

    def rename_current_speaker(self):
        if self.active or not self.current_dir or not self.speaker_select.currentData():
            return
        try:
            rename_speaker(self.current_dir, self.speaker_select.currentData(), self.speaker_rename.text())
            self.load_session(self.current_dir)
            self.status.setText('姓名已修改；请重新生成纪要后同步到 Notion。')
        except Exception as exc:
            self.handle_event('error', str(exc))

    def api_settings(self):
        return DeepSeekSettings(self.api_key.text().strip(), self.api_url.text().strip(),
                               self.api_model.text().strip(), self.api_timeout.value())

    def save_api(self):
        try:
            settings = self.api_settings()
            if settings.api_key:
                settings.validate()
            settings.save()
            self.status.setText('API 配置已保存到本机 .env')
        except Exception as exc:
            self.handle_event('error', str(exc))

    def config(self):
        if not self.output.text().strip() or not self.model.currentText().strip():
            raise ValueError('请填写输出目录和识别模型。')
        if self.interval.value() and not (self.full_screen.isChecked() or self.window.currentData()):
            raise ValueError('开启自动截图前，请先选择截图窗口。')
        return SessionConfig(out_dir=self.output.text().strip(),
            system_device=self.system_device.currentData(), mic_device=self.mic_device.currentData(),
            record_system=self.system_enabled.isChecked(), record_mic=self.mic_enabled.isChecked(),
            model_size=self.model.currentText().strip(), language=None if self.language.currentData()=='mixed' else self.language.currentData(),
            device=self.compute.currentData(), compute_type=self.precision.currentText(),
            cpu_threads=self.threads.value(), prompt=self.prompt.text().strip(),
            window_id=self.window.currentData(), full_screen=self.full_screen.isChecked(),
            shot_interval=max(5, self.interval.value()) if self.interval.value() else 0,
            diarize=self.diarize.isChecked(),participants=self.participants.value(),
            mic_is_participant=self.mic_is_participant.isChecked(),asr_backend=self.asr_backend.currentData(),
            multilingual=self.language.currentData()=='mixed',sensevoice_variant=self.sensevoice_variant.currentData())

    def _set_busy(self, busy, mode=''):
        self.start.setEnabled(not busy)
        for section in self.settings_tabs.sections:section.body.setEnabled(not busy)
        if busy and mode=='hardware':self.hardware_page.setEnabled(True)
        self.hardware_start.setEnabled(not busy)
        self.hardware_source.setEnabled(not busy)
        self.hardware_stop.setEnabled(busy and mode=='hardware')
        self.hardware_replay.setEnabled(not busy and self.hardware_file is not None)
        self.stop.setEnabled(busy and mode == 'record')
        self.shot.setEnabled(False)
        self.save_only.setEnabled(busy and mode in ('record', 'recover', 'summary', 'notion', 'speakers','hardware'))
        self.save_only.setText('停止测试并回放' if mode=='hardware' else ('取消任务' if mode in ('notion', 'speakers') else ('取消生成' if mode == 'summary' else ('取消恢复' if mode == 'recover' else '仅保存录音'))))
        self.summarize.setEnabled(not busy and self.current_dir is not None)
        self.recover.setEnabled(not busy and self.current_dir is not None)
        self.open_session.setEnabled(not busy)
        self.open_folder.setEnabled(self.current_dir is not None)
        self.sync_notion.setEnabled(not busy and self.current_dir is not None and (self.current_dir / 'minutes.md').is_file())

    def launch(self, function, result=None, mode=None):
        worker = TaskThread(function, self)
        self.workers.append(worker)
        worker.event.connect(self.handle_event)
        worker.failed.connect(lambda message: self.handle_event('error', message))
        if result:
            worker.result.connect(result)
        if mode:
            self.active = worker
            self._set_busy(True, mode)
        def finished():
            self.workers.remove(worker)
            if self.active is worker:
                self.active = None
                self._set_busy(False)
                if self.hotkeys:
                    self.hotkeys.close()
                    self.hotkeys = None
                self.session = None
            worker.deleteLater()
            if self.closing and not self.workers:
                self.close()
        worker.finished.connect(finished)
        worker.start()
        return worker

    def refresh_devices(self):
        self.refresh.setEnabled(False)
        def inventory(emit):
            devices, windows = [], []
            try:
                devices = list_devices()
            except Exception as exc:
                emit('warning', f'读取音频设备失败：{exc}')
            try:
                windows = list_windows()
            except Exception as exc:
                emit('warning', f'读取窗口列表失败：{exc}')
            return devices, windows
        def populated(result):
            devices, windows = result
            for combo, loopback, label in [(self.system_device, True, '系统默认扬声器'),
                                           (self.mic_device, False, '系统默认麦克风')]:
                selected = combo.currentData()
                combo.clear()
                combo.addItem(label, None)
                for device in devices:
                    if device.loopback == loopback:
                        combo.addItem(device.name, device.id)
                index = combo.findData(selected)
                combo.setCurrentIndex(max(0, index))
            selected = self.window.currentData()
            self.window.clear()
            self.window.addItem('请选择会议窗口', None)
            for hwnd, title in windows:
                if title != self.windowTitle():
                    self.window.addItem(title, hwnd)
            index = self.window.findData(selected)
            self.window.setCurrentIndex(max(0, index))
            self.refresh.setEnabled(True)
        self.launch(inventory, populated)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, '选择会议保存目录', self.output.text())
        if path:
            self.output.setText(path)

    def start_meeting(self):
        if self.active:
            return
        try:
            config = self.config()
            config.speaker_count()
            if config.asr_backend=='sensevoice' and not sensevoice_available(config.sensevoice_variant):
                raise ValueError('请先在转录模型设置中下载所选 SenseVoice 模型。')
            if not config.record_system and not config.record_mic:
                raise ValueError('请至少启用一路音频。')
            if self.auto_summary.isChecked():
                self.api_settings().validate()
            if config.diarize and config.record_system and not speaker_models_available():
                self.settings_tabs.setCurrentIndex(3)
                raise ValueError('请先在“说话人”设置中下载分析模型，或关闭说话人分析。')
        except Exception as exc:
            self.handle_event('error', str(exc))
            return
        self.transcript.clear()
        self.stop_playback()
        self.minutes.clear()
        self.shot_preview.setText('截图预览')
        self.notion_link.setText('尚未同步')
        self.cancel_task.clear()
        self.session = MeetingSession(config)
        current = self.session
        def run(emit):
            current.emit = emit
            return current.run()
        self.launch(run, self._record_finished, 'record')
        self.hotkeys = GlobalHotkeys(QApplication.instance(), self.keyshot, self.stop_meeting)
        for warning in self.hotkeys.install():
            self.handle_event('warning', warning)

    def _record_finished(self, path):
        self.load_session(path)
        if self.auto_summary.isChecked() and not self.closing and not (self.session and self.session.skip_transcription.is_set()):
            QTimer.singleShot(250, lambda: self._auto_generate(Path(path)))

    def _auto_generate(self, expected_root):
        if self.current_dir != expected_root:
            return
        if self.active:
            QTimer.singleShot(250, lambda: self._auto_generate(expected_root))
        elif not self.closing:
            self.generate_minutes()

    def stop_meeting(self):
        if self.session:
            self.session.stop()
            self.status.setText('正在结束会议并补完转录…')
            self.stop.setEnabled(False)
            self.shot.setEnabled(False)

    def save_audio_only(self):
        if self.session:
            self.session.stop(save_only=True)
        self.cancel_task.set()
        self.status.setText('正在取消后台处理并保存已有结果…')
        self.save_only.setEnabled(False)

    def keyshot(self):
        if self.session:
            self.session.request_screenshot()

    def handle_event(self, kind, value):
        if kind in ('status', 'warning', 'error'):
            if self.status.text() != str(value):
                self.status.setText(str(value))
                self.log.appendPlainText(f'[{datetime.now():%H:%M:%S}] {value}')
            if kind == 'error':
                self.tabs.setCurrentWidget(self.log)
        elif kind == 'session':
            self.current_dir = Path(value)
            self.session_path.setText(str(value))
            self.open_folder.setEnabled(True)
        elif kind == 'recording':
            self.shot.setEnabled(bool(value))
        elif kind == 'line':
            if self.current_dir and (self.current_dir / 'transcript.txt').exists():
                self.transcript.setPlainText((self.current_dir / 'transcript.txt').read_text(encoding='utf-8'))
                self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())
        elif kind == 'transcript_refresh':
            self.transcript.setPlainText((Path(value) / 'transcript.txt').read_text(encoding='utf-8'))
            self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())
        elif kind == 'speaker_progress':
            self.speaker_progress.setText(f'说话人已分析 {value:.0f} 秒 · CPU 并行运行')
        elif kind == 'hardware_progress':
            self.hardware_level.setValue(min(100,int(value['level']**.5*180)))
            self.hardware_result.setText(f'已录 {value["seconds"]:.1f} 秒 / 经过 {value["elapsed"]:.1f} 秒；说完后点击停止。')
        elif kind == 'metrics':
            for source, metric in value.items():
                self.levels[source].setValue(min(100, int(metric['level'] ** .5 * 180)))
                self.backlogs[source].setText(f'已录 {metric["seconds"]:.0f} 秒 · 待转录 {metric["backlog"]:.0f} 秒')
        elif kind == 'screenshot':
            pixmap = QPixmap(value.path)
            self.shot_preview.setPixmap(pixmap.scaled(700, 440, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.shot_preview.setToolTip(value.path)
            self.log.appendPlainText(f'重点截图已保存：{value.path}')

    def load_session(self, path):
        try:
            root = Path(path)
            data = recover_exports(root)
            self.speaker_select.clear()
            identities = {line.get('speaker'): line.get('speaker_name') for line in data['lines']
                          if (line.get('speaker') or '').startswith('SPEAKER_')}
            for speaker, name in sorted(identities.items()):
                self.speaker_select.addItem(f'{speaker} · {name or "未命名"}', speaker)
            self.notion_link.setText('可在 Notion 设置中同步当前纪要')
            self.current_dir = root
            self.session_path.setText(str(root))
            self.transcript.setPlainText((root / 'transcript.txt').read_text(encoding='utf-8'))
            self.minutes.clear()
            if (root / 'minutes.md').exists():
                self.minutes.setMarkdown((root / 'minutes.md').read_text(encoding='utf-8'))
            self.open_folder.setEnabled(True)
            if not self.active:
                self._set_busy(False)
        except Exception as exc:
            self.handle_event('error', f'打开会议失败：{exc}')

    def choose_session(self):
        path = QFileDialog.getExistingDirectory(self, '选择含 events.jsonl 的会议目录', self.output.text())
        if path:
            self.load_session(path)

    def generate_minutes(self):
        if self.active or not self.current_dir:
            return
        try:
            settings = self.api_settings()
            settings.validate()
        except Exception as exc:
            self.handle_event('error', str(exc))
            self.settings_tabs.setCurrentIndex(1)
            return
        self.cancel_task.clear()
        root = self.current_dir
        def run(emit):
            return DeepSeekSummarizer(settings, self.cancel_task,
                lambda message: emit('status', message)).generate(root)
        def done(markdown):
            self.minutes.setMarkdown(markdown)
            self.tabs.setCurrentWidget(self.minutes)
            self.status.setText('会议纪要已保存为 minutes.md')
            if self.notion_auto.isChecked() and not self.closing:
                QTimer.singleShot(250, lambda: self._auto_upload(root))
        self.launch(run, done, 'summary')

    def _auto_upload(self, expected_root):
        if self.current_dir != expected_root:
            return
        if self.active:
            QTimer.singleShot(250, lambda: self._auto_upload(expected_root))
        elif not self.closing:
            self.upload_notion()

    def recover_transcript(self):
        if self.active or not self.current_dir:
            return
        try:
            config = self.config()
        except Exception as exc:
            self.handle_event('error', str(exc))
            return
        root = self.current_dir
        self.cancel_task.clear()
        self.transcript.clear()
        self.launch(lambda emit: retranscribe(root, config, emit, self.cancel_task),
                    self.load_session, 'recover')

    def show_folder(self):
        if self.current_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_dir.resolve())))

    def closeEvent(self, event):
        self.stop_playback()
        if self.workers:
            self.closing = True
            if self.session:
                self.session.stop(save_only=True)
            self.cancel_task.set()
            self.status.setText('正在保存并关闭，请稍候…')
            event.ignore()
        else:
            if self.hotkeys:
                self.hotkeys.close()
            event.accept()


def run_gui():
    app = QApplication.instance() or QApplication([])
    app.setApplicationName('Meet Notes')
    window = MainWindow()
    window.show()
    return app.exec()
