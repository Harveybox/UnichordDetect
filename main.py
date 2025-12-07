import argparse
import signal
import sys
import threading
import time
from functools import partial

from audio_capture.audio_capture_wasapi_loopback import LoopbackCapture
from chord_estimator.chord_estimator import ChordEstimator
from timeline.timeline import Timeline
from ui.overlay_ui import run_overlay_app

DEFAULTS = {
    "sample_rate": 48000,
    "window_seconds": 1.5,
    "hop_seconds": 0.1,
    "smoothing_frames": 5,
    "min_confirm_seconds": 0.4,
    "ui_display_seconds": 90.0,
    "timeline_max_seconds": 120.0,
}


def run_app(device_index: int | None):
    capture = LoopbackCapture(device_index=device_index, target_sample_rate=DEFAULTS["sample_rate"])
    sample_rate, _channels = capture.start()

    timeline = Timeline(max_duration=DEFAULTS["timeline_max_seconds"])
    estimator = ChordEstimator(
        sample_rate=sample_rate,
        window_seconds=DEFAULTS["window_seconds"],
        hop_seconds=DEFAULTS["hop_seconds"],
        smoothing_frames=DEFAULTS["smoothing_frames"],
        min_confirm_seconds=DEFAULTS["min_confirm_seconds"],
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
    parser.add_argument("--list-devices", action="store_true", help="List loopback devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    run_app(args.device)


if __name__ == "__main__":
    main()
