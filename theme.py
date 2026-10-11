"""Consistent light controls even when Windows uses dark mode."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette


def apply_light_theme(app):
    if app is None:
        return
    app.setStyle('Fusion')
    if hasattr(app.styleHints(), 'setColorScheme'):
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    palette = QPalette()
    colors = {
        QPalette.Window: '#f5f7fb', QPalette.WindowText: '#23324b',
        QPalette.Base: '#ffffff', QPalette.AlternateBase: '#edf3fa',
        QPalette.Text: '#23324b', QPalette.Button: '#ffffff',
        QPalette.ButtonText: '#23324b', QPalette.BrightText: '#ffffff',
        QPalette.Highlight: '#2763bd', QPalette.HighlightedText: '#ffffff',
        QPalette.ToolTipBase: '#ffffff', QPalette.ToolTipText: '#23324b',
        QPalette.PlaceholderText: '#78849a', QPalette.Link: '#2763bd',
        QPalette.Light: '#ffffff', QPalette.Midlight: '#edf1f7',
        QPalette.Mid: '#d6dfec', QPalette.Dark: '#a8b4c6', QPalette.Shadow: '#78849a',
    }
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for role, color in colors.items():
            palette.setColor(group, role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor('#778398'))
    app.setPalette(palette)
