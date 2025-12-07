import sys
import time
from typing import List

from PySide6 import QtCore, QtGui, QtWidgets

from timeline.timeline import Segment


class TimelineWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, display_seconds: float = 60.0):
        super().__init__(parent)
        self.display_seconds = display_seconds
        self.segments: List[Segment] = []
        self.beats: List[float] = []
        self.setMinimumHeight(140)

    def update_segments(self, segments: List[Segment]):
        self.segments = segments
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 180))
        now = self.segments[-1].end if self.segments else time.time()
        width = self.width()
        height = self.height()
        # draw beat markers as thin vertical lines
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 1, QtCore.Qt.SolidLine))
        for b in self.beats:
            offset = max(0.0, now - b)
            if offset > self.display_seconds:
                continue
            x = width * (1 - offset / self.display_seconds)
            painter.drawLine(int(x), 0, int(x), height)

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
        font.setPointSize(28)
        font.setBold(True)
        self.label_display.setFont(font)
        self.label_display.setStyleSheet("color: white")
        header.addWidget(self.label_display)
        header.addStretch(1)

        # Controls area on left
        controls_widget = QtWidgets.QWidget()
        controls_widget.setFixedWidth(280)
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_widget.setLayout(controls_layout)

        title = QtWidgets.QLabel("Settings")
        title.setStyleSheet("color: white")
        controls_layout.addWidget(title)

        self.low_latency_cb = QtWidgets.QCheckBox("Low Latency")
        self.bass_focus_cb = QtWidgets.QCheckBox("Bass Focus")
        self.viterbi_cb = QtWidgets.QCheckBox("Use Viterbi")
        controls_layout.addWidget(self.low_latency_cb)
        controls_layout.addWidget(self.bass_focus_cb)
        controls_layout.addWidget(self.viterbi_cb)

        # display seconds control (scroll speed / zoom)
        self.display_seconds_sb = QtWidgets.QDoubleSpinBox()
        self.display_seconds_sb.setRange(5.0, 600.0)
        self.display_seconds_sb.setSuffix(" s")
        self.display_seconds_sb.setSingleStep(5.0)
        controls_layout.addWidget(QtWidgets.QLabel("Timeline Width:"))
        controls_layout.addWidget(self.display_seconds_sb)

        # bass param controls
        bass_inner = QtWidgets.QFormLayout()
        self.bass_low_sb = QtWidgets.QDoubleSpinBox()
        self.bass_low_sb.setRange(20.0, 300.0)
        self.bass_low_sb.setSuffix(" Hz")
        self.bass_high_sb = QtWidgets.QDoubleSpinBox()
        self.bass_high_sb.setRange(60.0, 1000.0)
        self.bass_high_sb.setSuffix(" Hz")
        self.bass_weight_sb = QtWidgets.QDoubleSpinBox()
        self.bass_weight_sb.setRange(0.1, 10.0)
        self.bass_weight_sb.setSingleStep(0.5)
        bass_inner.addRow("Bass Low:", self.bass_low_sb)
        bass_inner.addRow("Bass High:", self.bass_high_sb)
        bass_inner.addRow("Bass Weight:", self.bass_weight_sb)
        controls_layout.addLayout(bass_inner)

        # close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        controls_layout.addStretch(1)
        controls_layout.addWidget(close_button)

        layout.addLayout(header)
        # main content
        main_h = QtWidgets.QHBoxLayout()
        main_h.addWidget(controls_widget)
        main_h.addWidget(self.timeline_widget, 1)
        layout.addLayout(main_h)

        # callback to notify runtime
        self._on_settings_change = on_settings_change
        self._init_settings(initial_settings or {})
        # connect signals
        self.low_latency_cb.stateChanged.connect(self._emit_settings)
        self.bass_focus_cb.stateChanged.connect(self._emit_settings)
        self.viterbi_cb.stateChanged.connect(self._emit_settings)
        self.display_seconds_sb.valueChanged.connect(self._on_display_seconds_changed)
        self.bass_low_sb.valueChanged.connect(self._emit_settings)
        self.bass_high_sb.valueChanged.connect(self._emit_settings)
        self.bass_weight_sb.valueChanged.connect(self._emit_settings)

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
        self.viterbi_cb.setChecked(bool(s.get("use_viterbi", False)))
        self.display_seconds_sb.setValue(float(s.get("ui_display_seconds", 90.0)))
        # apply immediately
        try:
            self.timeline_widget.display_seconds = float(s.get("ui_display_seconds", 90.0))
        except Exception:
            pass

    def _emit_settings(self, _=None):
        if not self._on_settings_change:
            return
        s = {
            "low_latency": bool(self.low_latency_cb.isChecked()),
            "bass_focus": bool(self.bass_focus_cb.isChecked()),
            "use_viterbi": bool(self.viterbi_cb.isChecked()),
            "bass_low": float(self.bass_low_sb.value()),
            "bass_high": float(self.bass_high_sb.value()),
            "bass_weight": float(self.bass_weight_sb.value()),
        }
        try:
            self._on_settings_change(s)
        except Exception:
            # swallow exceptions from callback to avoid crashing UI
            pass

    def _on_display_seconds_changed(self, v: float):
        try:
            self.timeline_widget.display_seconds = float(v)
            self.timeline_widget.update()
        except Exception:
            pass

    def update_beats(self, beats: List[float]):
        self.timeline_widget.beats = beats
        self.timeline_widget.update()


def run_overlay_app(fetch_segments, refresh_ms: int = 30, display_seconds: float = 60.0, on_settings_change=None, initial_settings=None, fetch_beats=None):
    app = QtWidgets.QApplication(sys.argv)
    window = OverlayWindow(display_seconds=display_seconds, on_settings_change=on_settings_change, initial_settings=initial_settings)

    timer = QtCore.QTimer()
    def _tick():
        try:
            window.update_view(fetch_segments())
            if fetch_beats:
                window.update_beats(fetch_beats())
        except Exception:
            pass

    timer.timeout.connect(_tick)
    timer.start(refresh_ms)

    window.show()
    sys.exit(app.exec())


__all__ = ["run_overlay_app", "OverlayWindow"]
