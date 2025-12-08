"""
Soundcard-based loopback capture for Windows WASAPI.

This module provides an alternative to pyaudiowpatch using the 'soundcard' library,
which offers better cross-platform support and easier Windows WASAPI loopback access.
"""

import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class LoopbackDevice:
    def __init__(self, index: int, name: str, sample_rate: float, max_input_channels: int):
        self.index = index
        self.name = name
        self.sample_rate = sample_rate
        self.max_input_channels = max_input_channels

    def __repr__(self) -> str:
        return f"LoopbackDevice(index={self.index}, name={self.name}, rate={self.sample_rate}, channels={self.max_input_channels})"


class AudioRingBuffer:
    def __init__(self, max_frames: int):
        self.buffer: Deque[NDArray[np.int16]] = deque()
        self.size = 0
        self.max_frames = max_frames
        self.lock = threading.Lock()

    def push(self, frames: NDArray[np.int16]) -> None:
        with self.lock:
            self.buffer.append(frames)
            self.size += len(frames)
            while self.size > self.max_frames and self.buffer:
                dropped = self.buffer.popleft()
                self.size -= len(dropped)

    def pop(self, num_frames: int) -> Optional[NDArray[np.int16]]:
        with self.lock:
            if self.size < num_frames:
                return None
            out = []
            remaining = num_frames
            while remaining > 0 and self.buffer:
                chunk = self.buffer[0]
                if len(chunk) <= remaining:
                    out.append(chunk)
                    self.buffer.popleft()
                    remaining -= len(chunk)
                    self.size -= len(chunk)
                else:
                    out.append(chunk[:remaining])
                    self.buffer[0] = chunk[remaining:]
                    self.size -= remaining
                    remaining = 0
            if not out:
                return None
            return np.concatenate(out, axis=0)


class SoundcardLoopbackCapture:
    """
    Windows WASAPI loopback capture using the 'soundcard' library.
    Provides a fallback when pyaudiowpatch is unavailable.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        target_sample_rate: int = 48000,
        ring_seconds: int = 5,
    ):
        try:
            import soundcard
        except ImportError:
            raise RuntimeError("soundcard library is required but not installed.")

        self.soundcard = soundcard
        self.device_index = device_index
        self.target_sample_rate = target_sample_rate
        self.sample_rate = target_sample_rate
        self.channels = 2
        self.stream = None
        self.ring_buffer = AudioRingBuffer(max_frames=target_sample_rate * ring_seconds)
        self._stop = threading.Event()

    def list_loopback_devices(self) -> List[LoopbackDevice]:
        """List available loopback/speaker devices using soundcard."""
        devices: List[LoopbackDevice] = []
        try:
            # soundcard.all_microphones() includes loopback devices on Windows
            for i, mic in enumerate(self.soundcard.all_microphones(include_loopback=True)):
                if "stereo mix" in mic.name.lower() or "loopback" in mic.name.lower() or "what u hear" in mic.name.lower():
                    devices.append(
                        LoopbackDevice(
                            index=i,
                            name=mic.name,
                            sample_rate=self.target_sample_rate,
                            max_input_channels=2,
                        )
                    )
        except Exception:
            pass
        return devices

    def _select_device(self) -> tuple:
        """Select a loopback device or default microphone."""
        loopback_devices = self.list_loopback_devices()
        if loopback_devices:
            selected = loopback_devices[0]
            return self.soundcard.all_microphones(include_loopback=True)[selected.index], selected.sample_rate
        # Fallback to default microphone if no loopback found
        default_mic = self.soundcard.default_microphone()
        return default_mic, self.target_sample_rate

    def start(self) -> Tuple[int, int]:
        """Start recording from loopback device."""
        device, sr = self._select_device()
        self.sample_rate = int(sr)
        self.channels = 2

        def reader():
            """Background thread reading from device."""
            try:
                with device.recorder(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=1024,
                ) as rec:
                    while not self._stop.is_set():
                        try:
                            data = rec.record(numframes=1024)
                            # Convert float32 [-1, 1] to int16 [-32768, 32767]
                            int16_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
                            self.ring_buffer.push(int16_data)
                        except Exception:
                            time.sleep(0.01)
            except Exception as e:
                print(f"soundcard recording error: {e}")

        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()
        return self.sample_rate, self.channels

    def read(self, num_frames: int) -> Optional[NDArray[np.int16]]:
        """Read frames from ring buffer."""
        return self.ring_buffer.pop(num_frames)

    def stop(self) -> None:
        """Stop recording."""
        self._stop.set()
        if hasattr(self, "_reader_thread") and self._reader_thread:
            self._reader_thread.join(timeout=2)


__all__ = ["SoundcardLoopbackCapture", "LoopbackDevice", "AudioRingBuffer"]
