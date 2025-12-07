import os
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from chord_estimator.chord_estimator import ChordEstimate


class ChordinoStreamingEstimator:
    """
    Streaming wrapper around the VAMP Chordino plugin.
    It batches recent audio into a window, runs chordino, and emits the latest chord.

    Requirements:
    - Python module `vamp` (may need MSVC build tools on Windows).
    - Chordino VAMP plugin (e.g., nnls-chroma:chordino) available on VAMP_PATH.
    """

    def __init__(
        self,
        sample_rate: int,
        analysis_seconds: float = 8.0,
        hop_seconds: float = 2.0,
        target_sr: int = 44100,
        silence_rms: float = 1e-3,
        plugin_key: str = "nnls-chroma:chordino",
        vamp_path: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.analysis_seconds = analysis_seconds
        self.hop_seconds = hop_seconds
        self.target_sr = target_sr
        self.silence_rms = silence_rms
        self.plugin_key = plugin_key
        self.vamp_path = vamp_path

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffer: Deque[np.ndarray] = deque()
        self.on_estimate = None

        self._vamp = None
        self._librosa = None

    def start(self, ring_buffer, time_provider) -> None:
        try:
            import vamp
            import librosa
        except Exception as exc:  # pragma: no cover - runtime availability
            raise RuntimeError(
                "Chordino engine requires `vamp` and `librosa`. "
                "Install: pip install vamp librosa soundfile"
            ) from exc

        if self.vamp_path:
            os.environ["VAMP_PATH"] = self.vamp_path

        self._vamp = vamp
        self._librosa = librosa
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, args=(ring_buffer, time_provider), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _worker(self, ring_buffer, time_provider) -> None:
        hop_frames = max(1, int(self.hop_seconds * self.sample_rate))
        max_frames = int(self.analysis_seconds * self.sample_rate)
        last_infer = 0.0

        while not self._stop.is_set():
            chunk = ring_buffer.read(hop_frames)
            if chunk is None:
                time.sleep(0.05)
                continue

            self._buffer.append(chunk)
            total = sum(len(c) for c in self._buffer)
            while total > max_frames and self._buffer:
                dropped = self._buffer.popleft()
                total -= len(dropped)

            now = time_provider()
            if now - last_infer < self.hop_seconds:
                continue
            last_infer = now

            try:
                self._analyze(time_provider)
            except Exception as exc:
                print(f"chordino inference failed: {exc}")
                self._stop.set()

    def _analyze(self, time_provider) -> None:
        if not self._buffer:
            return

        audio = np.vstack(list(self._buffer))
        mono = audio.mean(axis=1).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(mono * mono)))
        if rms < self.silence_rms:
            if self.on_estimate:
                self.on_estimate(ChordEstimate("N", 0.0, time_provider()))
            return

        if self.sample_rate != self.target_sr:
            mono = self._librosa.resample(mono, orig_sr=self.sample_rate, target_sr=self.target_sr)
            sr = self.target_sr
        else:
            sr = self.sample_rate

        result = self._vamp.collect(mono, sr, self.plugin_key)
        segments = result.get("list", [])
        if not segments:
            return
        last_seg = segments[-1]
        label = last_seg.get("label", "N")
        ts = time_provider()
        if self.on_estimate:
            self.on_estimate(ChordEstimate(label, 1.0, ts))


__all__ = ["ChordinoStreamingEstimator"]
