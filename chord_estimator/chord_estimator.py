import threading
from collections import Counter, deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import math
import time


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
        # Bass-focus mode: emphasize frequencies in bass band and suppress higher melodic bands
        bass_focus: bool = False,
        bass_low: float = 50.0,
        bass_high: float = 350.0,
        bass_weight: float = 3.0,
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
        self.bass_focus = bass_focus
        self.bass_low = bass_low
        self.bass_high = bass_high
        self.bass_weight = bass_weight
        self.buffer: Deque[NDArray[np.int16]] = deque()
        # frame_queue stores (label, confidence) tuples for better smoothing decisions
        self.frame_queue: Deque[Tuple[str, float]] = deque(maxlen=smoothing_frames)
        self.lock = threading.Lock()
        self._last_confidence: float = 0.0
        self.templates, self.states = self._build_templates()
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.on_estimate = None  # callback receiving ChordEstimate
        self._last_label = "N"
        self._last_change_time: Optional[float] = None
        self._chroma_state: Optional[NDArray[np.float32]] = None
        self._viterbi_prev: Optional[NDArray[np.float32]] = None
        # onset / beat tracking for beat-synchronous gating
        self._onset_env_prev: Optional[NDArray[np.float32]] = None
        self._onset_times = deque(maxlen=32)
        self._beat_interval: Optional[float] = None
        # track low-frequency changes for downbeat/measure detection
        self._downbeat_env_prev: Optional[np.float32] = None
        # precompute transition log matrix for improved Viterbi
        self._trans_log = self._build_transition_log()

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

    def _build_transition_log(self) -> NDArray[np.float32]:
        # build transition log-probability matrix based on music theory and common progressions
        n = len(self.states)
        trans = np.full((n, n), -20.0, dtype=np.float32)  # very unlikely by default
        # allow staying in same state (high prob, avoid flutter)
        for i in range(n):
            trans[i, i] = math.log(0.7)  # boost stay probability to reduce flicker
        
        # musical knowledge: map chord roots and qualities
        chord_info = {}
        for i, s in enumerate(self.states):
            if s == "N":
                chord_info[i] = (None, None)  # silence
                continue
            parts = s.split(":")
            root = parts[0]
            quality = parts[1] if len(parts) > 1 else "maj"
            root_idx = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"].index(root)
            chord_info[i] = (root_idx, quality)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ri, qi = chord_info.get(i, (None, None))
                rj, qj = chord_info.get(j, (None, None))
                
                # transitions to/from silence
                if ri is None or rj is None:
                    trans[i, j] = math.log(0.05)
                    continue
                
                # interval in semitones (mod 12)
                interval = (rj - ri) % 12
                # common progressions: V->I, IV->I, IV->V, vi->IV, etc.
                # Likelihood boosted for musically "good" progressions
                
                prob = 0.01  # default very low
                
                # same root, different quality (major <-> minor)
                if interval == 0:
                    # prefer staying in same quality, but allow flip
                    prob = 0.2 if qi == qj else 0.08
                # strong resolutions (fifth)
                elif interval in (5, 7):  # V->I, IV->I dominant area
                    prob = 0.15 if qj == "maj" else 0.08
                # subdominant moves (whole step, major third, minor sixth)
                elif interval in (2, 4, 9):
                    prob = 0.10
                # chromatic/semitone moves
                elif interval == 1 or interval == 11:
                    prob = 0.04
                # tritone
                elif interval == 6:
                    prob = 0.03
                # other intervals
                else:
                    prob = 0.02
                
                # boost major-to-major progressions (more stable)
                if qi == "maj" and qj == "maj" and interval in (5, 7, 0, 2, 4):
                    prob *= 1.5
                
                trans[i, j] = math.log(max(prob, 1e-6))
        
        return trans

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
        while not self._stop.is_set():
            # read current hop/window under lock to avoid inconsistent mid-update values
            with self.lock:
                hop = int(self.hop_size)
                window_size = int(self.window_size)
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

            # use window_size captured under lock
            while current is not None and len(current) >= window_size:
                window = current[-window_size :]
                mono = window.mean(axis=1).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(mono * mono)))
                if rms < self.silence_rms:
                    self._reset_state_for_silence()
                    if self.on_estimate:
                        ts = time_provider()
                        self.on_estimate(ChordEstimate("N", 0.0, ts))
                    current = current[hop:]
                    continue

                # compute chroma and also track onset / beat info and bass energy
                chroma = self._compute_chroma_from_mono(mono, self.sample_rate)
                # update onset env and beat tracker (use real time epoch for UI)
                self._update_onset_and_beat(mono, self.sample_rate, time.time())
                # detect downbeats for measure-line visualization
                self._detect_downbeat(mono, self.sample_rate, time.time())
                # attempt bass root extraction
                bass_root, bass_conf = self._extract_bass_root(mono, self.sample_rate)
                label, confidence = self._match_chord(chroma, bass_root=bass_root, bass_conf=bass_conf)
                timestamp = time_provider()
                smoothed_label = self._smooth(label, confidence, timestamp)
                estimate = ChordEstimate(smoothed_label, confidence, timestamp)
                if self.on_estimate:
                    self.on_estimate(estimate)
                current = current[hop:]

    def get_onsets(self) -> List[float]:
        """Return a copy of recent onset times (seconds from epoch) for visualization."""
        with self.lock:
            return list(self._onset_times)

    def _compute_chroma_from_mono(self, mono: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        window = np.hanning(len(mono))
        spectrum = np.abs(np.fft.rfft(mono * window))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
        chroma = np.zeros(12, dtype=np.float32)
        # Iterate frequency bins and map to chroma with optional bass-focus weighting
        for mag, freq in zip(spectrum, freqs):
            # discard very low or very high energy to avoid bass rumble / ultrahigh hiss
            if freq < 20 or freq > self.hi_freq_cutoff:
                continue
            pc = int(round(12 * np.log2(freq / 440.0) + 69)) % 12
            # default weight behavior
            weight = 1.0
            # if bass_focus mode is enabled, strongly emphasize bass band and suppress higher bands
            if self.bass_focus:
                if freq < self.bass_low:
                    # ignore subsonic / extreme rumble
                    continue
                if freq <= self.bass_high:
                    weight = self.bass_weight
                else:
                    # attenuate bands above bass_high to reduce melody/vocal influence
                    weight = 0.1
            else:
                # legacy behavior: modest low-mid boost and high-frequency attenuation
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
        with self.lock:
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

    def _update_onset_and_beat(self, mono: NDArray[np.float32], sr: int, now: float) -> None:
        # quick spectral flux onset detector
        window = np.hanning(len(mono))
        spec = np.abs(np.fft.rfft(mono * window))
        # If previous onset env is missing or the FFT lengths changed (e.g. window/hop
        # updated for low-latency), reinitialize and skip a flux computation to
        # avoid shape mismatch errors.
        if self._onset_env_prev is None or self._onset_env_prev.shape != spec.shape:
            self._onset_env_prev = spec
            return
        flux = np.sum(np.maximum(spec - self._onset_env_prev, 0.0))
        self._onset_env_prev = spec
        # simple adaptive threshold
        thresh = 0.0005 * len(spec)
        if flux > thresh:
            # record onset time
            self._onset_times.append(now)
            if len(self._onset_times) >= 4:
                # compute median interval
                intervals = np.diff(np.array(self._onset_times))
                med = float(np.median(intervals)) if len(intervals) > 0 else None
                if med and 0.2 < med < 2.5:
                    self._beat_interval = med

    def _extract_bass_root(self, mono: NDArray[np.float32], sr: int) -> Tuple[Optional[int], float]:
        # find strongest spectral peak in bass band and convert to pitch-class
        n = len(mono)
        window = np.hanning(n)
        spec = np.abs(np.fft.rfft(mono * window))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        # mask bass band
        mask = (freqs >= max(20.0, self.bass_low)) & (freqs <= min(self.bass_high, self.hi_freq_cutoff))
        if not np.any(mask):
            return None, 0.0
        band = spec[mask]
        if band.sum() <= 0:
            return None, 0.0
        idx = int(np.argmax(band))
        freq_vals = freqs[mask]
        peak_freq = float(freq_vals[idx])
        # convert to midi note
        try:
            midi = 69 + 12 * math.log2(peak_freq / 440.0)
        except Exception:
            return None, 0.0
        pc = int(round(midi)) % 12
        conf = float(band[idx] / (band.sum() + 1e-9))
        return pc, conf

    def _match_chord(self, chroma: NDArray[np.float32], bass_root: Optional[int] = None, bass_conf: float = 0.0) -> Tuple[str, float]:
        scores = np.array([float(np.dot(chroma, self.templates[s])) for s in self.states], dtype=np.float32)
        energy = scores.sum()
        if energy < self.min_chroma_energy:
            return "N", 0.0
        # avoid all-zero; add tiny floor and normalize for emission
        scores = np.maximum(scores, 1e-6)
        emission = scores / scores.sum()

        # apply bass-root fusion: boost emission of candidates matching bass root
        if bass_root is not None and bass_conf > 0.02:
            for i, s in enumerate(self.states):
                if s == "N":
                    continue
                root = s.split(":")[0]
                root_idx = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"].index(root)
                if root_idx == bass_root:
                    emission[i] *= (1.0 + self.bass_weight * bass_conf)
            emission = emission / emission.sum()

        if self.use_viterbi:
            # improved Viterbi with precomputed transition log probabilities
            log_e = np.log(emission)
            if self._viterbi_prev is None:
                v_prev = log_e
            else:
                # v_new[j] = log_e[j] + max_i (v_prev[i] + trans_log[i,j])
                scores_mat = (self._viterbi_prev[:, None] + self._trans_log)
                v_prev = log_e + np.max(scores_mat, axis=0)
            self._viterbi_prev = v_prev
            best_idx = int(np.argmax(v_prev))
        else:
            self._viterbi_prev = None
            best_idx = int(np.argmax(emission))
        # confidence from emission: use top-3 separation for robustness
        # if top is much higher than both 2nd and 3rd, confidence is higher
        top3 = np.partition(emission, -3)[-3:]
        if top3.size >= 3:
            top_val = float(top3[-1])
            second_val = float(top3[-2])
            third_val = float(top3[-3])
            # margin penalizes closeness to 2nd; boost if 3rd is much lower (2-horse race)
            margin = (top_val - second_val) * (1.0 + 0.5 * (second_val - third_val) / (top_val + 1e-9))
        else:
            margin = float(top3[-1]) if top3.size > 0 else 0.0
        best_label = self.states[best_idx]
        return best_label, margin

    def _reset_state_for_silence(self) -> None:
        with self.lock:
            self.frame_queue.clear()
            self._chroma_state = None
            self._viterbi_prev = None
            self._last_label = "N"
            self._last_change_time = None

    def update_params(self, params: dict) -> None:
        """Update estimator parameters at runtime. Accepts a dict with keys
        like 'window_seconds', 'hop_seconds', 'smoothing_frames',
        'min_confirm_seconds', 'chroma_ema', and bass focus settings.
        This is thread-safe and will affect subsequent processing.
        """
        with self.lock:
            window_changed = False
            if "window_seconds" in params:
                self.window_size = int(params["window_seconds"] * self.sample_rate)
                window_changed = True
            if "hop_seconds" in params:
                self.hop_size = int(params["hop_seconds"] * self.sample_rate)
                window_changed = True
            if "smoothing_frames" in params:
                self.smoothing_frames = int(params["smoothing_frames"])
                # resize frame_queue maxlen while preserving entries
                self.frame_queue = deque(list(self.frame_queue), maxlen=self.smoothing_frames)
            if "min_confirm_seconds" in params:
                self.min_confirm_seconds = float(params["min_confirm_seconds"])
            if "chroma_ema" in params:
                self.chroma_ema = float(params["chroma_ema"])
            # bass-focus related
            if "bass_focus" in params:
                self.bass_focus = bool(params["bass_focus"])
            if "bass_low" in params:
                self.bass_low = float(params["bass_low"])
            if "bass_high" in params:
                self.bass_high = float(params["bass_high"])
            if "bass_weight" in params:
                self.bass_weight = float(params["bass_weight"])
            if "use_viterbi" in params:
                self.use_viterbi = bool(params["use_viterbi"])
            # If window/hop have changed, reset onset/beat state to avoid
            # FFT-length mismatches and stale timing assumptions.
            if window_changed:
                self._onset_env_prev = None
                self._downbeat_env_prev = None
                self._onset_times.clear()
                self._beat_interval = None

    def _smooth(self, label: str, confidence: float, timestamp: float) -> str:
        """Smoothing with confidence-aware hysteresis and consecutive frame requirement.

        frame_queue holds (label, confidence). We compute the majority label
        and the average confidence for that label. New labels must either be a
        fast change from silence or exceed a confidence threshold, be sustained
        per `min_confirm_seconds`, AND have at least ceil(smoothing_frames/2)
        consecutive matching frames before being committed.
        """
        with self.lock:
            self.frame_queue.append((label, float(confidence)))
            if len(self.frame_queue) > self.smoothing_frames:
                self.frame_queue.popleft()
            labels = [l for l, c in self.frame_queue]
            counter = Counter(labels)
            majority = counter.most_common(1)[0][0]
            # compute average confidence for majority label
            confidences = [c for l, c in self.frame_queue if l == majority]
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            # count consecutive frames matching majority from right
            consecutive = 0
            for i in range(len(labels) - 1, -1, -1):
                if labels[i] == majority:
                    consecutive += 1
                else:
                    break
            min_consecutive = max(2, len(self.frame_queue) // 2)

            # thresholds
            min_conf_threshold = 0.03

            if majority != self._last_label:
                # allow faster first change from silence/unknown if confidence reasonable
                if self._last_label == "N" and majority != "N" and avg_conf >= min_conf_threshold and consecutive >= 2:
                    self._last_change_time = timestamp
                    self._last_label = majority
                    self._last_confidence = avg_conf
                    return majority
                # require minimum duration before committing change
                if self._last_change_time is None:
                    self._last_change_time = timestamp
                    return self._last_label
                # require sustained duration
                if timestamp - self._last_change_time < self.min_confirm_seconds:
                    return self._last_label
                # require minimal confidence to avoid reacting to noise
                if avg_conf < min_conf_threshold:
                    return self._last_label
                # require confidence improvement over previous
                if avg_conf < (self._last_confidence + 0.02):
                    return self._last_label
                # require enough consecutive frames of majority label
                if consecutive < min_consecutive:
                    return self._last_label
                # commit
                self._last_change_time = timestamp
                self._last_label = majority
                self._last_confidence = avg_conf
                return majority

            # same as previous label: update timestamps/confidence
            self._last_change_time = timestamp
            self._last_confidence = avg_conf
            self._last_label = majority
            return majority


    def _detect_downbeat(self, mono: NDArray[np.float32], sr: int, now: float) -> None:
        """Detect downbeats (measure starts) based on low-frequency energy surge.
        Complements onset detection for better measure alignment.
        """
        # extract low-frequency band (40-200 Hz) for downbeat cues
        window = np.hanning(len(mono))
        spec = np.abs(np.fft.rfft(mono * window))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
        mask = (freqs >= 40.0) & (freqs <= 200.0)
        if np.any(mask):
            low_energy = float(np.mean(spec[mask]))
        else:
            low_energy = 0.0
        
        if self._downbeat_env_prev is None:
            self._downbeat_env_prev = low_energy
            return
        # simple peak detection: if low-freq energy rises significantly
        flux = low_energy - float(self._downbeat_env_prev)
        self._downbeat_env_prev = low_energy
        # adaptive threshold for downbeat detection
        thresh = 0.0003 * len(spec)
        if flux > thresh and self._beat_interval:
            # downbeat candidates are spaced by beat_interval * beats_per_bar
            self._onset_times.append(now)  # record as strong beat
    
    def get_measures(self, beats_per_bar: int = 4) -> List[float]:
        """Return inferred measure (bar) start times based on recent onsets.

        This is a heuristic: we group the recent onset times into measures by
        taking every `beats_per_bar`-th onset as a measure start. It provides a
        reasonable visual guideline for the UI to draw bar lines.
        """
        with self.lock:
            onsets = list(self._onset_times)
        if not onsets:
            return []
        return [onsets[i] for i in range(0, len(onsets), beats_per_bar)]


__all__ = ["ChordEstimator", "ChordEstimate"]
