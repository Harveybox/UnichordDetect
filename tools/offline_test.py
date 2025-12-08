#!/usr/bin/env python3
"""Offline test harness: feed a WAV file through the ChordEstimator logic
without audio backend. Useful to reproduce 'always N' behavior locally.

Usage:
  python tools/offline_test.py path/to/file.wav

Only supports 16-bit PCM WAV files (stereo or mono).
"""
import sys
import wave
import numpy as np
import time
from chord_estimator.chord_estimator import ChordEstimator, ChordEstimate


def read_wav(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.getnframes()
        raw = w.readframes(frames)
    if sampwidth != 2:
        raise RuntimeError('Only 16-bit WAV supported')
    data = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    return sr, data


class Printer:
    def __call__(self, estimate: ChordEstimate):
        ts = estimate.timestamp
        print(f"{ts:.3f}: {estimate.label} ({estimate.confidence:.3f})")


def run_offline(path):
    sr, data = read_wav(path)
    print(f"Loaded {path} sr={sr} frames={len(data)} channels={data.shape[1]}")
    # choose window/hop similar to main defaults
    est = ChordEstimator(sample_rate=sr, window_seconds=0.7, hop_seconds=0.08, smoothing_frames=4, min_confirm_seconds=0.5)
    est.on_estimate = Printer()

    window = est.window_size
    hop = est.hop_size
    # iterate over data
    t0 = time.time()
    idx = 0
    while idx + window <= len(data):
        block = data[idx: idx + window]
        mono = block.mean(axis=1).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(mono * mono)))
        if rms < est.silence_rms:
            est._reset_state_for_silence()
            est.on_estimate(ChordEstimate('N', 0.0, time.time() - t0))
            idx += hop
            continue
        chroma = est._compute_chroma_from_mono(mono, sr)
        bass_root, bass_conf = est._extract_bass_root(mono, sr)
        label, confidence = est._match_chord(chroma, bass_root=bass_root, bass_conf=bass_conf)
        ts = time.time() - t0
        sm = est._smooth(label, float(confidence), ts)
        est.on_estimate(ChordEstimate(sm, confidence, ts))
        idx += hop


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/offline_test.py file.wav')
        sys.exit(1)
    run_offline(sys.argv[1])
