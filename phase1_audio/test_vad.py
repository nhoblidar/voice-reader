"""Tests with synthetic audio, so the VAD is verifiable without a microphone.
Run:  pytest -v
"""
import numpy as np

from vad import FRAME_SIZE, SAMPLE_RATE, EnergyVAD

rng = np.random.default_rng(0)


def noise(seconds: float, amp: float = 0.003) -> np.ndarray:
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * amp).astype(np.float32)


def tone(seconds: float, amp: float = 0.2, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32) + noise(seconds)


def run(vad: EnergyVAD, audio: np.ndarray):
    events = []
    for i in range(0, len(audio) - FRAME_SIZE + 1, FRAME_SIZE):
        events += vad.process(audio[i:i + FRAME_SIZE])
    return events


def test_detects_one_utterance_with_accurate_timing():
    audio = np.concatenate([noise(1.0), tone(1.0), noise(1.5)])
    events = run(EnergyVAD(), audio)
    assert [e.kind for e in events] == ["start", "end"]
    assert abs(events[0].time_s - 1.0) < 0.05
    assert abs(events[1].time_s - 2.0) < 0.05


def test_ignores_a_short_click():
    click = tone(0.02)  # one frame: shorter than the attack window
    audio = np.concatenate([noise(1.0), click, noise(1.0)])
    assert run(EnergyVAD(), audio) == []


def test_short_pause_does_not_split_the_utterance():
    audio = np.concatenate([noise(1.0), tone(0.5), noise(0.2), tone(0.5), noise(1.5)])
    assert [e.kind for e in run(EnergyVAD(), audio)] == ["start", "end"]


def test_utterance_includes_preroll():
    audio = np.concatenate([noise(1.0), tone(1.0), noise(1.5)])
    end = run(EnergyVAD(), audio)[-1]
    # 1.0s speech + 0.2s pre-roll + 0.5s release tail, give or take a frame
    assert len(end.audio) / SAMPLE_RATE > 1.6


def test_adapts_to_a_louder_room():
    quiet_then_loud_room = np.concatenate([noise(1.0), noise(3.0, amp=0.02)])
    events = run(EnergyVAD(noise_adapt=0.2), quiet_then_loud_room)
    # a step up in background noise may trigger once, but must not stay "speaking"
    assert len(events) <= 2
