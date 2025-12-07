import os
import tempfile
import threading
import time
from typing import Deque, Optional

import numpy as np
from numpy.typing import NDArray

from chord_estimator.chord_estimator import ChordEstimate


class AutoChordStreamingEstimator:
    """
    Wraps the autochord GitHub project for streaming use by periodically
    dumping a sliding window of audio to a temp WAV and invoking
    autochord.recognize on it.

    Notes:
    - autochord imports will attempt to download a TensorFlow model via gdown
      and copy the NNLS-Chroma VAMP plugin; this can take time and requires
      internet + a compatible VAMP binary (the upstream repo ships a Linux .so).
    - This estimator is best-effort: if autochord is missing or fails to load,
      it will raise at start so the caller can fall back.
    """

    def __init__(
        self,
        sample_rate: int,
        analysis_seconds: float = 12.0,
        hop_seconds: float = 3.0,
        target_sr: int = 44100,
    ):
        self.sample_rate = sample_rate
        self.analysis_seconds = analysis_seconds
        self.hop_seconds = hop_seconds
        self.target_sr = target_sr
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffer: Deque[NDArray[np.int16]] = None  # type: ignore
        self._autochord = None
        self._librosa = None
        self._wavfile = None
        self.on_estimate = None

    def start(self, ring_buffer, time_provider) -> None:
        try:
            import autochord
            import librosa
            from scipy.io import wavfile
        except Exception as exc:  # pragma: no cover - runtime availability
            raise RuntimeError(
                "autochord is not available. Install via "
                "`pip install git+https://github.com/cjbayron/autochord` "
                "and ensure its dependencies (tensorflow, vamp, gdown, librosa) are satisfied."
            ) from exc

        # stash imports so worker does not re-import
        from collections import deque

        self._autochord = autochord
        self._librosa = librosa
        self._wavfile = wavfile
        self._buffer = deque()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, args=(ring_buffer, time_provider), daemon=True
        )
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
            # keep only the latest analysis window
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
                # surface once and stop to avoid log spam
                print(f"autochord inference failed: {exc}")
                self._stop.set()

    def _analyze(self, time_provider) -> None:
        if not self._buffer:
            return

        audio = np.vstack(list(self._buffer))
        mono = audio.mean(axis=1).astype(np.float32) / 32768.0

        if self.sample_rate != self.target_sr:
            mono = self._librosa.resample(mono, orig_sr=self.sample_rate, target_sr=self.target_sr)
            sr = self.target_sr
        else:
            sr = self.sample_rate

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
        # write 16-bit PCM for autochord
        self._wavfile.write(tmp_path, sr, np.clip(mono, -1.0, 1.0).astype(np.float32))

        try:
            labels = self._autochord.recognize(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if not labels:
            return

        _start, _end, chord_name = labels[-1]
        ts = time_provider()
        if self.on_estimate:
            self.on_estimate(ChordEstimate(chord_name, 1.0, ts))


__all__ = ["AutoChordStreamingEstimator"]
