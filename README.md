# voice-reader

A voice-controlled reading engine, built from scratch to understand every layer of real-time voice AI.
It reads any document aloud, and you can interrupt it naturally: "stop", "go back", "what does that mean?"

## Roadmap

- [x] **Phase 1: Audio I/O + VAD from scratch**
- [ ] Phase 2: Streaming TTS playback with word-level position tracking
- [ ] Phase 3: Barge-in and echo cancellation (hand-built NLMS filter)
- [ ] Phase 4: Streaming STT + benchmark against Silero VAD
- [ ] Phase 5: Intent routing: commands vs questions about the document
- [ ] Phase 6: Latency instrumentation + evaluation harness
- [ ] Phase 7: WebRTC + browser extension
- [ ] Phase 8: Train a small turn-detection model

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd phase1_audio
pytest -v             # verify the VAD with synthetic audio
python live_mic.py    # try it live; stay quiet for half a second first
```

On Linux you may need `sudo apt install libportaudio2`. On macOS, allow microphone access for your terminal.

## Phase 1: what you're learning

**Digital audio.** A microphone produces a stream of numbers (samples). At 16 kHz you get 16,000 per second,
each a float between -1 and 1. We process them in 20 ms frames of 320 samples.

**Loudness in dBFS.** RMS measures a frame's energy; converting to decibels makes it logarithmic like hearing.
0 dBFS is the loudest possible signal; a quiet room is typically -60 to -45.

**The VAD state machine.** Two states (quiet, speech) with asymmetric transitions: 60 ms of loudness to start,
500 ms of quiet to end. The asymmetry is deliberate: false starts are cheap to ignore, but ending too early
cuts people off mid-sentence.

**Real-time threading.** The audio driver calls our callback on a real-time thread that must never block.
It only enqueues samples; all processing happens on the main thread.

## Experiments to run (and write down results)

1. Change `--margin` to 6 and 20. What breaks at each extreme?
2. Set `release_frames` to 10 (200 ms). Read a long sentence slowly. What happens?
3. Turn on a fan or play music. Watch the floor adapt. Where does energy-based VAD fail?
4. Whisper instead of talking. Why does it struggle? (Hint: whispers have little low-frequency energy.)

## Interview talking points

- Why energy VAD fails in noise, and why neural VADs (Silero) and semantic turn detectors exist
- The latency/accuracy trade-off of the release window: every ms of hangover adds directly to response latency
- Why the noise floor only adapts on quiet frames
- Why pre-roll matters for STT accuracy
- Why audio callbacks must never block
