"""
Live microphone demo: shows a loudness meter, prints speech start/end events,
and saves each utterance to a .wav file so you can listen to what the VAD caught.

Run:  python live_mic.py            (Ctrl+C to stop)
      python live_mic.py --margin 8 (more sensitive)
"""
from __future__ import annotations

import argparse
import queue
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from vad import FRAME_SIZE, SAMPLE_RATE, EnergyVAD


def save_wav(path: Path, audio: np.ndarray) -> None:
    """float32 in [-1, 1] -> 16-bit PCM WAV (the most universal audio format)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 2 bytes = 16 bits
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def meter(level_db: float, threshold_db: float, width: int = 40) -> str:
    """Map -80..0 dBFS onto a text bar, with '|' marking the speech threshold."""
    def pos(db: float) -> int:
        return int(np.clip((db + 80) / 80, 0, 1) * (width - 1))
    bar = ["#" if i <= pos(level_db) else " " for i in range(width)]
    bar[pos(threshold_db)] = "|"
    return "".join(bar)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=12.0, help="dB above noise floor")
    parser.add_argument("--out", type=Path, default=Path("utterances"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    # The audio callback runs on a real-time thread owned by the sound driver.
    # It must NEVER block (no printing, no disk, no heavy math) or you get
    # dropouts. So it only copies the samples into a queue; the main thread
    # does all the actual work. This producer/consumer split is the core
    # pattern of every real-time audio system.
    frames: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, n_frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        frames.put(indata[:, 0].copy())

    vad = EnergyVAD(margin_db=args.margin)
    pending = np.zeros(0, dtype=np.float32)
    count = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE,
                        channels=1, dtype="float32", callback=callback):
        print("Listening... stay quiet for a moment while I calibrate. Ctrl+C to stop.\n")
        try:
            while True:
                # Re-frame defensively: some drivers deliver odd block sizes.
                pending = np.concatenate([pending, frames.get()])
                while len(pending) >= FRAME_SIZE:
                    frame, pending = pending[:FRAME_SIZE], pending[FRAME_SIZE:]
                    for ev in vad.process(frame):
                        if ev.kind == "start":
                            print(f"\n>>> speech START at {ev.time_s:6.2f}s")
                        else:
                            count += 1
                            path = args.out / f"utt_{count:03d}.wav"
                            save_wav(path, ev.audio)
                            dur = len(ev.audio) / SAMPLE_RATE
                            print(f"\n<<< speech END   at {ev.time_s:6.2f}s  ({dur:.2f}s saved to {path})")

                    state = "CALIBRATING" if vad.calibrating else ("SPEECH" if vad.in_speech else "quiet ")
                    print(f"\r[{meter(vad.last_level_db, vad.threshold_db)}] "
                          f"{vad.last_level_db:6.1f} dB  floor {vad.noise_db:6.1f}  {state}   ",
                          end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
