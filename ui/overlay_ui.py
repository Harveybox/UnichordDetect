import sys
from typing import List

from PySide6 import QtCore, QtGui, QtWidgets

from timeline.timeline import Segment


class TimelineWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, display_seconds: float = 60.0):
        super().__init__(parent)
        self.display_seconds = display_seconds
        self.segments: List[Segment] = []
        self.setMinimumHeight(120)

    def update_segments(self, segments: List[Segment]):
        self.segments = segments
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 180))
        if not self.segments:
            return
        now = self.segments[-1].end
        width = self.width()
        height = self.height()
        for seg in self.segments:
            start_offset = max(0.0, now - seg.end)
            end_offset = max(0.0, now - seg.start)
            if end_offset > self.display_seconds:
                continue
            x1 = width * (1 - end_offset / self.display_seconds)
            x2 = width * (1 - start_offset / self.display_seconds)
            rect = QtCore.QRectF(x1, 0, x2 - x1, height)
            color = self._color_for_label(seg.label)
            painter.fillRect(rect, color)
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(rect, QtCore.Qt.AlignCenter, f"{seg.label}\n{seg.confidence:.2f}")

    @staticmethod
    def _color_for_label(label: str) -> QtGui.QColor:
        if label == "N":
            return QtGui.QColor(80, 80, 80, 200)
        colors = [
            QtGui.QColor("#e6194b"),
            QtGui.QColor("#3cb44b"),
            QtGui.QColor("#ffe119"),
            QtGui.QColor("#0082c8"),
            QtGui.QColor("#f58231"),
            QtGui.QColor("#911eb4"),
            QtGui.QColor("#46f0f0"),
            QtGui.QColor("#f032e6"),
            QtGui.QColor("#d2f53c"),
            QtGui.QColor("#fabebe"),
            QtGui.QColor("#008080"),
            QtGui.QColor("#e6beff"),
        ]
        idx = hash(label) % len(colors)
        color = colors[idx]
        color.setAlpha(220)
        return color


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, display_seconds: float = 60.0, on_settings_change=None, initial_settings=None):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowTitle("Chord Timeline")
        self.resize(900, 200)
        self.timeline_widget = TimelineWidget(display_seconds=display_seconds)
        self.current_label = "N"
        self.current_confidence = 0.0
        self._drag_pos = None
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(layout)

        header = QtWidgets.QHBoxLayout()
        self.label_display = QtWidgets.QLabel("N")
        font = QtGui.QFont()
        font.setPointSize(32)
        font.setBold(True)
        self.label_display.setFont(font)
        self.label_display.setStyleSheet("color: white")
        header.addWidget(self.label_display)
        header.addStretch(1)
        # Controls area: settings for debugging
        controls = QtWidgets.QHBoxLayout()
        self.low_latency_cb = QtWidgets.QCheckBox("低延迟")
        self.bass_focus_cb = QtWidgets.QCheckBox("低频聚焦")
        # spinboxes for bass band
        self.bass_low_sb = QtWidgets.QDoubleSpinBox()
        self.bass_low_sb.setPrefix("低频低端:")
        self.bass_low_sb.setSuffix(" Hz")
        self.bass_low_sb.setRange(20.0, 300.0)
        self.bass_low_sb.setSingleStep(10.0)
        self.bass_high_sb = QtWidgets.QDoubleSpinBox()
        self.bass_high_sb.setPrefix("低频高端:")
        self.bass_high_sb.setSuffix(" Hz")
        self.bass_high_sb.setRange(60.0, 1000.0)
        self.bass_high_sb.setSingleStep(10.0)
        self.bass_weight_sb = QtWidgets.QDoubleSpinBox()
        self.bass_weight_sb.setPrefix("权重:")
        self.bass_weight_sb.setSingleStep(0.5)
        self.bass_weight_sb.setRange(0.1, 10.0)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)

        controls.addWidget(self.low_latency_cb)
        controls.addWidget(self.bass_focus_cb)
        controls.addWidget(self.bass_low_sb)
        controls.addWidget(self.bass_high_sb)
        controls.addWidget(self.bass_weight_sb)
        controls.addStretch(1)
        controls.addWidget(close_button)
        header.addLayout(controls)
        # callback to notify runtime
        self._on_settings_change = on_settings_change
        self._init_settings(initial_settings or {})
        # connect signals
        self.low_latency_cb.stateChanged.connect(self._emit_settings)
        self.bass_focus_cb.stateChanged.connect(self._emit_settings)
        self.bass_low_sb.valueChanged.connect(self._emit_settings)
        self.bass_high_sb.valueChanged.connect(self._emit_settings)
        self.bass_weight_sb.valueChanged.connect(self._emit_settings)
        layout.addLayout(header)

        layout.addWidget(self.timeline_widget)

    def update_view(self, segments: List[Segment]):
        self.timeline_widget.update_segments(segments)
        if segments:
            current = segments[-1]
            self.label_display.setText(f"{current.label} ({current.confidence:.2f})")

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._drag_pos is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self._drag_pos = None

    def _init_settings(self, s: dict):
        # initialize UI controls from settings dict
        self.low_latency_cb.setChecked(bool(s.get("low_latency", False)))
        self.bass_focus_cb.setChecked(bool(s.get("bass_focus", False)))
        self.bass_low_sb.setValue(float(s.get("bass_low", 50.0)))
        self.bass_high_sb.setValue(float(s.get("bass_high", 350.0)))
        self.bass_weight_sb.setValue(float(s.get("bass_weight", 3.0)))

    def _emit_settings(self, _=None):
        if not self._on_settings_change:
            return
        s = {
            "low_latency": bool(self.low_latency_cb.isChecked()),
            "bass_focus": bool(self.bass_focus_cb.isChecked()),
            "bass_low": float(self.bass_low_sb.value()),
            "bass_high": float(self.bass_high_sb.value()),
            "bass_weight": float(self.bass_weight_sb.value()),
        }
        try:
            self._on_settings_change(s)
        except Exception:
            # swallow exceptions from callback to avoid crashing UI
            pass


def run_overlay_app(fetch_segments, refresh_ms: int = 30, display_seconds: float = 60.0, on_settings_change=None, initial_settings=None):
    app = QtWidgets.QApplication(sys.argv)
    window = OverlayWindow(display_seconds=display_seconds, on_settings_change=on_settings_change, initial_settings=initial_settings)

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: window.update_view(fetch_segments()))
    timer.start(refresh_ms)

    window.show()
    sys.exit(app.exec())


__all__ = ["run_overlay_app", "OverlayWindow"]
