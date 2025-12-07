import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

try:
    import pyaudiowpatch as pyaudio  # preferred WASAPI build
except ImportError:
    try:
        import pyaudio  # fallback if user installed plain PyAudio
    except ImportError:  # pragma: no cover - runtime dependency
        pyaudio = None


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
        self.buffer: Deque[np.ndarray] = deque()
        self.size = 0
        self.max_frames = max_frames
        self.lock = threading.Lock()

    def push(self, frames: np.ndarray) -> None:
        with self.lock:
            self.buffer.append(frames)
            self.size += len(frames)
            while self.size > self.max_frames and self.buffer:
                dropped = self.buffer.popleft()
                self.size -= len(dropped)

    def pop(self, num_frames: int) -> Optional[np.ndarray]:
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


class LoopbackCapture:
    def __init__(self, device_index: Optional[int] = None, target_sample_rate: int = 48000, ring_seconds: int = 5):
        if pyaudio is None:
            raise RuntimeError("pyaudiowpatch is required for WASAPI loopback")
        self.pa = pyaudio.PyAudio()
        self.device_index = device_index
        self.target_sample_rate = target_sample_rate
        self.sample_rate = target_sample_rate
        self.channels = 2
        self.stream = None
        self.ring_buffer = AudioRingBuffer(max_frames=target_sample_rate * ring_seconds)
        self._stop = threading.Event()

    def list_loopback_devices(self) -> List[LoopbackDevice]:
        devices: List[LoopbackDevice] = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info.get("isLoopbackDevice"):
                devices.append(
                    LoopbackDevice(
                        index=i,
                        name=info.get("name"),
                        sample_rate=info.get("defaultSampleRate", self.target_sample_rate),
                        max_input_channels=info.get("maxInputChannels", 2),
                    )
                )
        return devices

    def _select_device(self) -> LoopbackDevice:
        if self.device_index is not None:
            info = self.pa.get_device_info_by_index(self.device_index)
            return LoopbackDevice(
                index=self.device_index,
                name=info.get("name"),
                sample_rate=info.get("defaultSampleRate", self.target_sample_rate),
                max_input_channels=info.get("maxInputChannels", 2),
            )
        # pick default loopback
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info.get("isLoopbackDevice") and info.get("name", "").lower().startswith("default"):
                return LoopbackDevice(
                    index=i,
                    name=info.get("name"),
                    sample_rate=info.get("defaultSampleRate", self.target_sample_rate),
                    max_input_channels=info.get("maxInputChannels", 2),
                )
        devices = self.list_loopback_devices()
        if not devices:
            raise RuntimeError("No WASAPI loopback devices found")
        return devices[0]

    def start(self) -> Tuple[int, int]:
        device = self._select_device()
        self.sample_rate = int(device.sample_rate)
        self.channels = min(device.max_input_channels or 2, 2)
        frames_per_buffer = 1024

        def callback(in_data, frame_count, time_info, status):
            if self._stop.is_set():
                return (None, pyaudio.paComplete)
            if in_data:
                data = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self.channels)
                self.ring_buffer.push(data)
            return (None, pyaudio.paContinue)

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
            input_device_index=device.index,
            stream_callback=callback,
        )
        self.stream.start_stream()
        return self.sample_rate, self.channels

    def read(self, num_frames: int) -> Optional[np.ndarray]:
        return self.ring_buffer.pop(num_frames)

    def stop(self) -> None:
        self._stop.set()
        if self.stream is not None:
            while self.stream.is_active():
                time.sleep(0.05)
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


__all__ = ["LoopbackCapture", "LoopbackDevice", "AudioRingBuffer"]
