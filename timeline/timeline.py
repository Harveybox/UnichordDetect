import threading
from dataclasses import dataclass
from typing import List
from collections import deque


@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float


class Timeline:
    def __init__(self, max_duration: float = 120.0, min_segment_duration: float = 0.25):
        # use deque for efficient popleft when trimming old segments
        self.segments: deque[Segment] = deque()
        self.max_duration = max_duration
        self.lock = threading.Lock()
        # collapse any very short segments (seconds) to reduce UI jitter
        self.min_segment_duration = min_segment_duration

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
                # compact any very short segments to avoid rapid flicker in UI
                self._compact_short_segments()

    def _trim(self, now: float) -> None:
        cutoff = now - self.max_duration
        # efficiently pop from left while segments are older than cutoff
        while self.segments and self.segments[0].end < cutoff:
            self.segments.popleft()

    def _compact_short_segments(self) -> None:
        """Merge segments shorter than `min_segment_duration` into their neighbors
        to reduce rapid flicker when labels change briefly.
        """
        if not self.segments:
            return
        merged: List[Segment] = []
        for seg in list(self.segments):
            dur = seg.end - seg.start
            if dur < self.min_segment_duration and merged:
                # merge into previous segment
                prev = merged[-1]
                prev.end = seg.end
                # optionally bump confidence toward the more confident
                prev.confidence = max(prev.confidence, seg.confidence)
            else:
                merged.append(Segment(seg.start, seg.end, seg.label, seg.confidence))
        # replace deque with merged list
        self.segments = deque(merged)

    def get_recent(self) -> List[Segment]:
        with self.lock:
            # return a shallow copy as list for UI consumption
            return [Segment(s.start, s.end, s.label, s.confidence) for s in list(self.segments)]


__all__ = ["Timeline", "Segment"]
