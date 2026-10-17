"""Local neural VAD and bounded phrase segmentation, independent of audio I/O."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioSegment:
    samples: np.ndarray
    started_at: float
    ended_at: float = 0.0
    reason: str = "pause"


class SileroVAD:
    """16 kHz / 512 sample ONNX contract, with recurrent state and 64 sample context.

    Model interface: snakers4/silero-vad, utils_vad.OnnxWrapper (MIT).
    No Torch dependency and no model download during application startup.
    """
    frame_samples = 512

    def __init__(self, path: Path):
        import onnxruntime as ort
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD modeli eksik: {path}. README kurulumuna bakın.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, 64), np.float32)

    def __call__(self, frame):
        if frame.shape != (512,):
            raise ValueError("Silero VAD 512 örnek / 16 kHz bekliyor")
        audio = np.concatenate((self.context, frame.reshape(1, -1)), axis=1)
        probability, self.state = self.session.run(None, {
            "input": audio, "state": self.state, "sr": np.array(16000, dtype=np.int64)})
        self.context = audio[:, -64:].copy()
        return float(probability.reshape(-1)[0])


class SpeechSegmenter:
    """Hysteresis avoids splitting weak syllables; bounded context avoids runaway buffers.

    Prefer a recent short pause for long utterances. If none exists, wait up to
    hard_limit seconds, then split explicitly. This does not claim semantic endpoints.
    """
    rate = 16000
    frame_samples = 512

    def __init__(self, detector, silence_ms=384, max_seconds=6.5, hard_seconds=8.0):
        self.detector = detector
        self.silence_frames = max(1, round(silence_ms / 32))
        self.max_frames = round(max_seconds / .032)
        self.hard_frames = round(hard_seconds / .032)
        self.reset()

    def reset(self):
        self.detector.reset()
        self.preroll = deque(maxlen=6)
        self.frames = []
        self.speech_frames = 0
        self.silence = 0
        self.active = False
        self.started_at = 0.0
        self.last_voice_end = 0.0

    def feed(self, frame, started_at):
        probability = self.detector(frame)
        voiced = probability >= (0.35 if self.active else 0.5)
        frame_end = started_at + len(frame) / self.rate
        if not self.active:
            if not voiced:
                self.preroll.append((frame.copy(), started_at))
                return None
            self.frames = [item[0] for item in self.preroll]
            self.started_at = self.preroll[0][1] if self.preroll else started_at
            self.preroll.clear()
            self.active = True

        self.frames.append(frame.copy())
        if voiced:
            self.speech_frames += 1
            self.last_voice_end = frame_end
            self.silence = 0
        else:
            self.silence += 1

        # A real pause after a long phrase permits an earlier endpoint. Keep at
        # least 256 ms so short intra-word dropouts do not fragment the phrase.
        pause_frames = 8 if len(self.frames) >= 64 else self.silence_frames
        reason = None
        if self.silence >= pause_frames:
            reason = "pause"
        elif len(self.frames) >= self.max_frames and self.silence >= 3:
            reason = "long_pause"
        elif len(self.frames) >= self.hard_frames:
            reason = "limit"
        if reason is None:
            return None

        # Keep a small acoustic tail, not the whole endpoint waiting interval.
        tail = self.frames[-6:]
        trim = max(0, self.silence - 3)
        selected = self.frames[:-trim] if trim else self.frames
        segment = None
        if self.speech_frames >= 8:
            segment = AudioSegment(np.concatenate(selected), self.started_at,
                                   self.last_voice_end, reason)
        # Do not repeat speech at a forced boundary. Silent pre-roll can be reused.
        self.frames = []
        self.speech_frames = self.silence = 0
        self.active = False
        self.preroll.clear()
        if reason != "limit":
            tail_start = frame_end - len(tail) * .032
            self.preroll.extend((part, tail_start+i*.032) for i, part in enumerate(tail))
        return segment
