import threading
from collections import Counter, deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class ChordEstimate:
    def __init__(self, label: str, confidence: float, timestamp: float):
        self.label = label
        self.confidence = confidence
        self.timestamp = timestamp


class ChordEstimator:
    def __init__(
        self,
        sample_rate: int,
        window_seconds: float = 1.5,
        hop_seconds: float = 0.1,
        smoothing_frames: int = 5,
        min_confirm_seconds: float = 0.4,
    ):
        self.sample_rate = sample_rate
        self.window_size = int(window_seconds * sample_rate)
        self.hop_size = int(hop_seconds * sample_rate)
        self.smoothing_frames = smoothing_frames
        self.min_confirm_seconds = min_confirm_seconds
        self.buffer: Deque[NDArray[np.int16]] = deque()
        self.frame_queue: Deque[ChordEstimate] = deque(maxlen=smoothing_frames)
        self.lock = threading.Lock()
        self.templates = self._build_templates()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.on_estimate = None  # callback receiving ChordEstimate
        self._last_label = "N"
        self._last_change_time: Optional[float] = None

    @staticmethod
    def _build_templates() -> dict:
        templates = {}
        # major and minor triads across 12 pitch classes
        for i, name in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]):
            vec = np.zeros(12)
            vec[i] = 1  # root
            vec[(i + 4) % 12] = 0.8  # major third
            vec[(i + 7) % 12] = 0.9  # fifth
            templates[f"{name}:maj"] = vec
            vec_min = np.zeros(12)
            vec_min[i] = 1
            vec_min[(i + 3) % 12] = 0.8
            vec_min[(i + 7) % 12] = 0.9
            templates[f"{name}:min"] = vec_min
        templates["N"] = np.zeros(12)
        return templates

    def start(self, ring_buffer, time_provider) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, args=(ring_buffer, time_provider), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _worker(self, ring_buffer, time_provider) -> None:
        current = np.zeros((0, 2), dtype=np.int16)
        hop = self.hop_size
        while not self._stop.is_set():
            needed = hop
            chunk = ring_buffer.read(needed)
            if chunk is None:
                threading.Event().wait(0.01)
                continue
            current = np.vstack([current, chunk])
            while len(current) >= self.window_size:
                window = current[-self.window_size :]
                chroma = self._compute_chroma(window, self.sample_rate)
                label, confidence = self._match_chord(chroma)
                timestamp = time_provider()
                smoothed_label = self._smooth(label, timestamp)
                estimate = ChordEstimate(smoothed_label, confidence, timestamp)
                if self.on_estimate:
                    self.on_estimate(estimate)
                current = current[hop:]

    def _compute_chroma(self, frames: NDArray[np.int16], sr: int) -> NDArray[np.float32]:
        mono = frames.mean(axis=1).astype(np.float32)
        mono = mono / 32768.0
        window = np.hanning(len(mono))
        spectrum = np.abs(np.fft.rfft(mono * window))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
        chroma = np.zeros(12, dtype=np.float32)
        for mag, freq in zip(spectrum, freqs):
            if freq < 50 or freq > 5000:
                continue
            pc = int(round(12 * np.log2(freq / 440.0) + 69)) % 12
            chroma[pc] += mag
        if chroma.sum() > 0:
            chroma /= chroma.sum()
        return chroma

    def _match_chord(self, chroma: NDArray[np.float32]) -> Tuple[str, float]:
        best_label = "N"
        best_score = -1.0
        second_score = -1.0
        for label, tmpl in self.templates.items():
            if label == "N":
                continue
            score = float(np.dot(chroma, tmpl))
            if score > best_score:
                second_score = best_score
                best_score = score
                best_label = label
            elif score > second_score:
                second_score = score
        confidence = max(0.0, best_score - second_score) if second_score >= 0 else max(0.0, best_score)
        if best_score < 0.01:
            return "N", 0.0
        return best_label, confidence

    def _smooth(self, label: str, timestamp: float) -> str:
        self.frame_queue.append(label)
        if len(self.frame_queue) > self.smoothing_frames:
            self.frame_queue.popleft()
        counter = Counter(self.frame_queue)
        majority = counter.most_common(1)[0][0]
        if majority != self._last_label:
            # require minimum duration before committing change
            if self._last_change_time is None:
                self._last_change_time = timestamp
                return self._last_label
            if timestamp - self._last_change_time < self.min_confirm_seconds:
                return self._last_label
            self._last_change_time = timestamp
            self._last_label = majority
            return majority
        self._last_change_time = timestamp
        self._last_label = majority
        return majority


__all__ = ["ChordEstimator", "ChordEstimate"]
