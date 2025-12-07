import threading
from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float


class Timeline:
    def __init__(self, max_duration: float = 120.0):
        self.segments: List[Segment] = []
        self.max_duration = max_duration
        self.lock = threading.Lock()

    def update(self, timestamp: float, label: str, confidence: float) -> None:
        with self.lock:
            if not self.segments:
                self.segments.append(Segment(timestamp, timestamp, label, confidence))
                return
            current = self.segments[-1]
            if label == current.label:
                current.end = timestamp
            else:
                current.end = timestamp
                self.segments.append(Segment(timestamp, timestamp, label, confidence))
            self._trim(timestamp)

    def _trim(self, now: float) -> None:
        cutoff = now - self.max_duration
        while self.segments and self.segments[0].end < cutoff:
            self.segments.pop(0)

    def get_recent(self) -> List[Segment]:
        with self.lock:
            return [Segment(s.start, s.end, s.label, s.confidence) for s in self.segments]


__all__ = ["Timeline", "Segment"]
