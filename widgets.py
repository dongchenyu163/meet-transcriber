"""Collapsible settings sections with keyboard-accessible headers."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget,QVBoxLayout,QToolButton,QScrollArea,QSizePolicy


class CollapsibleSection(QWidget):
    def __init__(self,title,content,expanded=False,parent=None):
        super().__init__(parent)
        layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.header=QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.header.setObjectName('sectionHeader')
        self.body=content
        layout.addWidget(self.header)
        layout.addWidget(content)
        self.header.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)

    def set_expanded(self,expanded):
        self.header.setChecked(expanded)
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)


class SettingsSections(QScrollArea):
    """Preserve logical section indices used by validation/navigation."""
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.container=QWidget()
        self.column=QVBoxLayout(self.container)
        self.column.setContentsMargins(5,5,5,5)
        self.column.setSpacing(6)
        self.column.addStretch()
        self.setWidget(self.container)
        self.sections=[]
        self.selected=0

    def addTab(self,content,title):
        if isinstance(content,QScrollArea):
            old=content
            content=old.takeWidget()
            old.deleteLater()
        section=CollapsibleSection(title,content,expanded=not self.sections)
        self.column.insertWidget(len(self.sections),section)
        self.sections.append(section)
        section.header.clicked.connect(lambda checked,i=len(self.sections)-1: setattr(self,'selected',i))

    def setCurrentIndex(self,index):
        self.selected=index
        section=self.sections[index]
        section.set_expanded(True)
        # Bring the section's contents into view as well as its header.
        QTimer.singleShot(0,lambda:self.verticalScrollBar().setValue(section.y()))

    def currentIndex(self):return self.selected
