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
    def __init__(self, display_seconds: float = 60.0):
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
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        header.addWidget(close_button)
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


def run_overlay_app(fetch_segments, refresh_ms: int = 30, display_seconds: float = 60.0):
    app = QtWidgets.QApplication(sys.argv)
    window = OverlayWindow(display_seconds=display_seconds)

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: window.update_view(fetch_segments()))
    timer.start(refresh_ms)

    window.show()
    sys.exit(app.exec())


__all__ = ["run_overlay_app", "OverlayWindow"]
