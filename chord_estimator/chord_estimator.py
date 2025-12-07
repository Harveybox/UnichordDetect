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
        smoothing_frames: int = 7,
        min_confirm_seconds: float = 0.8,
        chroma_ema: float = 0.7,
        viterbi_switch_penalty: float = 0.8,
        min_chroma_energy: float = 1e-4,
        low_freq_boost: float = 2.0,
        hi_freq_cutoff: float = 1800.0,
        silence_rms: float = 1e-3,
        use_viterbi: bool = False,
        high_freq_attenuation: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.window_size = int(window_seconds * sample_rate)
        self.hop_size = int(hop_seconds * sample_rate)
        self.smoothing_frames = smoothing_frames
        self.min_confirm_seconds = min_confirm_seconds
        self.chroma_ema = chroma_ema
        self.viterbi_switch_penalty = viterbi_switch_penalty
        self.min_chroma_energy = min_chroma_energy
        self.low_freq_boost = low_freq_boost
        self.hi_freq_cutoff = hi_freq_cutoff
        self.silence_rms = silence_rms
        self.use_viterbi = use_viterbi
        self.high_freq_attenuation = high_freq_attenuation
        self.buffer: Deque[NDArray[np.int16]] = deque()
        self.frame_queue: Deque[ChordEstimate] = deque(maxlen=smoothing_frames)
        self.lock = threading.Lock()
        self.templates, self.states = self._build_templates()
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.on_estimate = None  # callback receiving ChordEstimate
        self._last_label = "N"
        self._last_change_time: Optional[float] = None
        self._chroma_state: Optional[NDArray[np.float32]] = None
        self._viterbi_prev: Optional[NDArray[np.float32]] = None

    @staticmethod
    def _build_templates() -> Tuple[dict, List[str]]:
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
        states = list(templates.keys())
        return templates, states

    def start(self, ring_buffer, time_provider) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, args=(ring_buffer, time_provider), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _worker(self, ring_buffer, time_provider) -> None:
        current: NDArray[np.int16] | None = None
        hop = self.hop_size
        while not self._stop.is_set():
            needed = hop
            chunk = ring_buffer.read(needed)
            if chunk is None:
                threading.Event().wait(0.01)
                continue
            if current is None:
                current = chunk
            else:
                if current.shape[1] != chunk.shape[1]:
                    current = chunk
                else:
                    current = np.vstack([current, chunk])

            while current is not None and len(current) >= self.window_size:
                window = current[-self.window_size :]
                mono = window.mean(axis=1).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(mono * mono)))
                if rms < self.silence_rms:
                    self._reset_state_for_silence()
                    if self.on_estimate:
                        ts = time_provider()
                        self.on_estimate(ChordEstimate("N", 0.0, ts))
                    current = current[hop:]
                    continue

                chroma = self._compute_chroma_from_mono(mono, self.sample_rate)
                label, confidence = self._match_chord(chroma)
                timestamp = time_provider()
                smoothed_label = self._smooth(label, timestamp)
                estimate = ChordEstimate(smoothed_label, confidence, timestamp)
                if self.on_estimate:
                    self.on_estimate(estimate)
                current = current[hop:]

    def _compute_chroma_from_mono(self, mono: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        window = np.hanning(len(mono))
        spectrum = np.abs(np.fft.rfft(mono * window))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
        chroma = np.zeros(12, dtype=np.float32)
        for mag, freq in zip(spectrum, freqs):
            # discard very low or very high energy to avoid bass rumble / ultrahigh hiss
            if freq < 50 or freq > self.hi_freq_cutoff:
                continue
            pc = int(round(12 * np.log2(freq / 440.0) + 69)) % 12
            # emphasize low-mid (bass instruments) to reduce vocal dominance
            if freq < 350:
                weight = self.low_freq_boost
            elif freq > 800:
                weight = self.high_freq_attenuation
            else:
                weight = 1.0
            chroma[pc] += mag * weight
        if chroma.sum() > 0:
            chroma /= chroma.sum()
        # exponential moving average to stabilize chroma against transient melodies
        if self._chroma_state is None:
            self._chroma_state = chroma
        else:
            self._chroma_state = (
                self.chroma_ema * self._chroma_state + (1 - self.chroma_ema) * chroma
            )
        smoothed = self._chroma_state.copy()
        if smoothed.sum() > 0:
            smoothed /= smoothed.sum()
        return smoothed

    def _match_chord(self, chroma: NDArray[np.float32]) -> Tuple[str, float]:
        scores = np.array([float(np.dot(chroma, self.templates[s])) for s in self.states], dtype=np.float32)
        energy = scores.sum()
        if energy < self.min_chroma_energy:
            return "N", 0.0
        # avoid all-zero; add tiny floor and normalize for emission
        scores = np.maximum(scores, 1e-6)
        emission = scores / scores.sum()

        if self.use_viterbi:
            # Viterbi smoothing: prefer staying in the same chord unless evidence is strong
            if self._viterbi_prev is None:
                v_prev = np.log(emission)
            else:
                # slight decay to avoid hard lock into one state
                stay_scores = self._viterbi_prev - 0.05  # decay over time
                switch_best = np.max(self._viterbi_prev) - self.viterbi_switch_penalty
                # combine stay vs switch
                combined = np.maximum(stay_scores, switch_best)
                v_prev = combined + np.log(emission)
            self._viterbi_prev = v_prev
            best_idx = int(np.argmax(v_prev))
        else:
            self._viterbi_prev = None
            best_idx = int(np.argmax(emission))
        # confidence from emission margin (top vs second)
        top2 = np.partition(emission, -2)[-2:]
        margin = float(top2[-1] - top2[-2]) if top2.size == 2 else float(top2[-1])
        best_label = self.states[best_idx]
        return best_label, margin

    def _reset_state_for_silence(self) -> None:
        self.frame_queue.clear()
        self._chroma_state = None
        self._viterbi_prev = None
        self._last_label = "N"
        self._last_change_time = None

    def _smooth(self, label: str, timestamp: float) -> str:
        self.frame_queue.append(label)
        if len(self.frame_queue) > self.smoothing_frames:
            self.frame_queue.popleft()
        counter = Counter(self.frame_queue)
        majority = counter.most_common(1)[0][0]
        if majority != self._last_label:
            # allow faster first change from silence/unknown
            if self._last_label == "N" and majority != "N":
                self._last_change_time = timestamp
                self._last_label = majority
                return majority
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
