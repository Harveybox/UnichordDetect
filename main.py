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
    "bass_focus": False,
    "bass_low": 50.0,
    "bass_high": 350.0,
    "bass_weight": 3.0,
}

LOW_LATENCY = {
    "sample_rate": 48000,
    "window_seconds": 0.4,  # smaller window for faster response (~200ms latency)
    "hop_seconds": 0.05,  # faster analysis cycles (~50ms between decisions)
    "smoothing_frames": 2,  # minimal smoothing, accept more jitter for speed
    "min_confirm_seconds": 0.2,  # quick confirmation (200ms min hold time)
    "chroma_ema": 0.7,  # slightly higher EMA for stability despite smaller window
    "viterbi_switch_penalty": 0.0,  # keep Viterbi off
    "ring_seconds": 5,  # smaller ring buffer (5s instead of 20s)
    "ui_display_seconds": 90.0,
    "timeline_max_seconds": 120.0,
    "autochord_analysis_seconds": 12.0,
    "autochord_hop_seconds": 3.0,
    "bass_focus": True,
    "bass_low": 50.0,
    "bass_high": 350.0,
    "bass_weight": 3.0,
}


def run_app(
    device_index: int | None,
    engine: str,
    vamp_path: str | None,
    fallback_input: bool = False,
    low_latency: bool = False,
    bass_focus: bool = False,
    bass_low: float | None = None,
    bass_high: float | None = None,
    bass_weight: float | None = None,
):
    params = LOW_LATENCY if low_latency else DEFAULTS
    # apply CLI overrides for bass options if provided
    if bass_focus:
        params = dict(params)  # shallow copy
        params["bass_focus"] = True
    if bass_low is not None:
        params = dict(params)
        params["bass_low"] = bass_low
    if bass_high is not None:
        params = dict(params)
        params["bass_high"] = bass_high
    if bass_weight is not None:
        params = dict(params)
        params["bass_weight"] = bass_weight
    capture = LoopbackCapture(
        device_index=device_index,
        target_sample_rate=params["sample_rate"],
        ring_seconds=params["ring_seconds"],
        allow_fallback=fallback_input,
    )
    sample_rate, _channels = capture.start()

    timeline = Timeline(max_duration=params["timeline_max_seconds"])
    if engine == "autochord":
        estimator = AutoChordStreamingEstimator(
            sample_rate=sample_rate,
            analysis_seconds=params["autochord_analysis_seconds"],
            hop_seconds=params["autochord_hop_seconds"],
            target_sr=44100,
        )
    elif engine == "chordino":
        estimator = ChordinoStreamingEstimator(
            sample_rate=sample_rate,
            analysis_seconds=params["autochord_analysis_seconds"],
            hop_seconds=params["autochord_hop_seconds"],
            target_sr=44100,
            vamp_path=vamp_path,
        )
    else:
        estimator = ChordEstimator(
            sample_rate=sample_rate,
            window_seconds=params["window_seconds"],
            hop_seconds=params["hop_seconds"],
            smoothing_frames=params["smoothing_frames"],
            min_confirm_seconds=params["min_confirm_seconds"],
            chroma_ema=params["chroma_ema"],
            viterbi_switch_penalty=params["viterbi_switch_penalty"],
            min_chroma_energy=1e-5,
            low_freq_boost=3.0,
            hi_freq_cutoff=1200.0,
            high_freq_attenuation=0.25,
            bass_focus=params.get("bass_focus", False),
            bass_low=params.get("bass_low", 50.0),
            bass_high=params.get("bass_high", 350.0),
            bass_weight=params.get("bass_weight", 3.0),
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
    fetch_beats = lambda: estimator.get_onsets() if hasattr(estimator, 'get_onsets') else []

    # callback from UI when settings change
    def on_settings_change(settings: dict):
        # settings may include low_latency and bass focus; map to estimator update
        try:
            # if low_latency toggled, pick base preset
            if "low_latency" in settings:
                preset = LOW_LATENCY if settings.get("low_latency") else DEFAULTS
                # copy bass settings from UI into preset
                preset = dict(preset)
                preset["bass_focus"] = settings.get("bass_focus", preset.get("bass_focus", False))
                preset["bass_low"] = settings.get("bass_low", preset.get("bass_low", 50.0))
                preset["bass_high"] = settings.get("bass_high", preset.get("bass_high", 350.0))
                preset["bass_weight"] = settings.get("bass_weight", preset.get("bass_weight", 3.0))
                estimator.update_params(preset)
            else:
                # only bass parameters changed
                estimator.update_params(settings)
        except Exception:
            pass

    try:
        run_overlay_app(
            fetch_segments,
            display_seconds=params["ui_display_seconds"],
            on_settings_change=on_settings_change,
            initial_settings={
                "low_latency": True if low_latency else False,
                "bass_focus": params.get("bass_focus", False),
                "bass_low": params.get("bass_low", 50.0),
                "bass_high": params.get("bass_high", 350.0),
                "bass_weight": params.get("bass_weight", 3.0),
                "use_viterbi": params.get("use_viterbi", False),
            },
            fetch_beats=fetch_beats,
        )
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
    parser.add_argument(
        "--low-latency",
        action="store_true",
        help="Use low-latency mode (target <0.5s latency, trade off some accuracy)",
    )
    parser.add_argument(
        "--bass-focus",
        action="store_true",
        help="Enable bass-focus mode: emphasize low-frequency band for chord detection",
    )
    parser.add_argument("--bass-low", type=float, default=None, help="Bass low cutoff (Hz)")
    parser.add_argument("--bass-high", type=float, default=None, help="Bass high cutoff (Hz)")
    parser.add_argument("--bass-weight", type=float, default=None, help="Weight multiplier for bass band")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # override params from CLI if provided
    # pass bass options through to run_app via flags; run_app will read params presets
    # apply CLI overrides into selected params dict inside run_app by passing low_latency and bass flags
    # for simplicity, forward values as arguments
    # run
    run_app(
        args.device,
        args.engine,
        args.vamp_path,
        fallback_input=args.fallback_input,
        low_latency=args.low_latency,
        bass_focus=args.bass_focus,
        bass_low=args.bass_low,
        bass_high=args.bass_high,
        bass_weight=args.bass_weight,
    )


if __name__ == "__main__":
    main()
