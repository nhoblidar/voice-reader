"""
Energy-based Voice Activity Detector, built from scratch.

The idea: speech is louder than the background. We measure the loudness of
each 20 ms frame, keep a running estimate of the room's noise floor, and call
a frame "speech" when it is clearly louder than that floor.

Four tricks turn that naive idea into something usable:
  1. Calibration   - learn the room's noise level before detecting anything.
  2. Adaptive floor - keep updating the noise estimate, but ONLY from quiet
                      frames, so speech doesn't teach us that speech is noise.
  3. Hysteresis    - require several loud frames to START (attack) and many
                      quiet frames to END (release/hangover). This stops a
                      single click from triggering, and stops natural pauses
                      between words from ending the utterance.
  4. Pre-roll      - keep the last few frames before speech starts, so the
                      first syllable isn't clipped off (STT hates that).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000  # 16 kHz is the standard rate for speech recognition
FRAME_MS = 20         # 20 ms frames: short enough to react fast, long enough to measure
FRAME_SIZE = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples per frame


def rms_dbfs(frame: np.ndarray) -> float:
    """Loudness of one frame in dB relative to full scale (0 dBFS = maximum).

    RMS (root mean square) measures average energy. Converting to decibels
    makes it logarithmic, like human hearing: every -6 dB halves the amplitude.
    """
    rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
    return 20.0 * np.log10(max(rms, 1e-10))  # clamp so silence isn't -inf


@dataclass
class VADEvent:
    kind: str                          # "start" or "end"
    time_s: float                      # position in the stream, in seconds
    audio: np.ndarray | None = None    # the full utterance, attached to "end"


class EnergyVAD:
    def __init__(
        self,
        margin_db: float = 12.0,         # how far above the noise floor counts as speech
        min_threshold_db: float = -55.0, # never treat anything quieter than this as speech 
        attack_frames: int = 3,          # 60 ms of loudness to start
        release_frames: int = 25,        # 500 ms of quiet to end
        preroll_frames: int = 10,        # keep 200 ms before speech starts
        calibration_frames: int = 25,    # 500 ms to learn the room
        noise_adapt: float = 0.05,       # how fast the floor tracks changes (0..1)
    ):
        self.margin_db = margin_db
        self.min_threshold_db = min_threshold_db
        self.attack_frames = attack_frames
        self.release_frames = release_frames
        self.calibration_frames = calibration_frames
        self.noise_adapt = noise_adapt

        self.noise_db = -60.0
        self.in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._frames_seen = 0
        self._calib_levels: list[float] = []
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self._utterance: list[np.ndarray] = []
        self.last_level_db = -200.0

    @property
    def threshold_db(self) -> float:
        return max(self.min_threshold_db, self.noise_db + self.margin_db)

    @property
    def calibrating(self) -> bool:
        return self._frames_seen < self.calibration_frames

    def process(self, frame: np.ndarray) -> list[VADEvent]:
        """Feed exactly one frame. Returns any events it triggered."""
        if len(frame) != FRAME_SIZE:
            raise ValueError(f"expected {FRAME_SIZE} samples, got {len(frame)}")

        level = rms_dbfs(frame)
        t = self._frames_seen * FRAME_MS / 1000
        self._frames_seen += 1
        self.last_level_db = level

        # 1. Calibration: just listen and average.
        if self._frames_seen <= self.calibration_frames:
            self._calib_levels.append(level)
            self.noise_db = float(np.mean(self._calib_levels))
            self._preroll.append(frame)
            return []

        is_loud = level > self.threshold_db
        events: list[VADEvent] = []

        if not self.in_speech:
            self._preroll.append(frame)
            if is_loud:
                self._speech_run += 1
            else:
                self._speech_run = 0
                # 2. Adaptive floor: exponential moving average of quiet frames.
                self.noise_db += self.noise_adapt * (level - self.noise_db)

            # 3a. Attack: enough consecutive loud frames -> speech started.
            if self._speech_run >= self.attack_frames:
                self.in_speech = True
                self._silence_run = 0
                self._utterance = list(self._preroll)  # 4. include pre-roll
                self._preroll.clear()
                start_t = t - (self.attack_frames - 1) * FRAME_MS / 1000
                events.append(VADEvent("start", start_t))
        else:
            self._utterance.append(frame)
            self._silence_run = 0 if is_loud else self._silence_run + 1

            # 3b. Release: enough consecutive quiet frames -> speech ended.
            if self._silence_run >= self.release_frames:
                self.in_speech = False
                self._speech_run = 0
                audio = np.concatenate(self._utterance)
                self._utterance = []
                end_t = t - (self.release_frames - 1) * FRAME_MS / 1000
                events.append(VADEvent("end", end_t, audio))

        return events
