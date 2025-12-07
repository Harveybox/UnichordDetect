import argparse
import signal
import sys
import threading
import time
from functools import partial

from audio_capture.audio_capture_wasapi_loopback import LoopbackCapture
from chord_estimator.autochord_streaming import AutoChordStreamingEstimator
from chord_estimator.chord_estimator import ChordEstimator
from chord_estimator.chordino_streaming import ChordinoStreamingEstimator
from timeline.timeline import Timeline
from ui.overlay_ui import run_overlay_app

DEFAULTS = {
    "sample_rate": 48000,
    "window_seconds": 0.7,  # balance speed and stability
    "hop_seconds": 0.08,  # responsive hop
    "smoothing_frames": 4,  # moderate smoothing to resist melody
    "min_confirm_seconds": 0.5,  # faster than bar-length but avoids flicker
    "chroma_ema": 0.6,  # moderate EMA
    "viterbi_switch_penalty": 0.0,  # keep Viterbi off
    "ring_seconds": 20,
    "ui_display_seconds": 90.0,
    "timeline_max_seconds": 120.0,
    "autochord_analysis_seconds": 12.0,
    "autochord_hop_seconds": 3.0,
}


def run_app(device_index: int | None, engine: str, vamp_path: str | None, fallback_input: bool = False):
    capture = LoopbackCapture(
        device_index=device_index,
        target_sample_rate=DEFAULTS["sample_rate"],
        ring_seconds=DEFAULTS["ring_seconds"],
        allow_fallback=fallback_input,
    )
    sample_rate, _channels = capture.start()

    timeline = Timeline(max_duration=DEFAULTS["timeline_max_seconds"])
    if engine == "autochord":
        estimator = AutoChordStreamingEstimator(
            sample_rate=sample_rate,
            analysis_seconds=DEFAULTS["autochord_analysis_seconds"],
            hop_seconds=DEFAULTS["autochord_hop_seconds"],
            target_sr=44100,
        )
    elif engine == "chordino":
        estimator = ChordinoStreamingEstimator(
            sample_rate=sample_rate,
            analysis_seconds=DEFAULTS["autochord_analysis_seconds"],
            hop_seconds=DEFAULTS["autochord_hop_seconds"],
            target_sr=44100,
            vamp_path=vamp_path,
        )
    else:
        estimator = ChordEstimator(
            sample_rate=sample_rate,
            window_seconds=DEFAULTS["window_seconds"],
            hop_seconds=DEFAULTS["hop_seconds"],
            smoothing_frames=DEFAULTS["smoothing_frames"],
            min_confirm_seconds=DEFAULTS["min_confirm_seconds"],
            chroma_ema=DEFAULTS["chroma_ema"],
            viterbi_switch_penalty=DEFAULTS["viterbi_switch_penalty"],
            min_chroma_energy=1e-5,
            low_freq_boost=3.0,
            hi_freq_cutoff=1200.0,
            high_freq_attenuation=0.25,
            use_viterbi=False,
        )

    stop_event = threading.Event()

    def time_provider():
        return time.perf_counter()

    def handle_estimate(estimate):
        timeline.update(estimate.timestamp, estimate.label, estimate.confidence)

    estimator.on_estimate = handle_estimate
    estimator.start(capture, time_provider)

    def stop_all():
        if stop_event.is_set():
            return
        stop_event.set()
        estimator.stop()
        capture.stop()

    signal.signal(signal.SIGINT, lambda sig, frame: stop_all())
    signal.signal(signal.SIGTERM, lambda sig, frame: stop_all())

    fetch_segments = lambda: timeline.get_recent()
    try:
        run_overlay_app(fetch_segments, display_seconds=DEFAULTS["ui_display_seconds"])
    finally:
        stop_all()


def list_devices():
    capture = LoopbackCapture()
    devices = capture.list_loopback_devices()
    print("Available loopback devices:")
    for d in devices:
        print(f"[{d.index}] {d.name} (rate={d.sample_rate}, channels={d.max_input_channels})")


def main():
    parser = argparse.ArgumentParser(description="Universal chord recognition overlay")
    parser.add_argument("--device", type=int, default=None, help="WASAPI loopback device index")
    parser.add_argument(
        "--fallback-input",
        action="store_true",
        help="When no WASAPI loopback device is found, fall back to the default input device (for testing).",
    )
    parser.add_argument("--list-devices", action="store_true", help="List loopback devices and exit")
    parser.add_argument(
        "--engine",
        choices=["simple", "autochord", "chordino"],
        default="simple",
        help="Chord estimator engine (default: simple)",
    )
    parser.add_argument(
        "--vamp-path",
        type=str,
        default=None,
        help="Path to VAMP plugins (set if using chordino engine)",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    run_app(args.device, args.engine, args.vamp_path, fallback_input=args.fallback_input)


if __name__ == "__main__":
    main()
