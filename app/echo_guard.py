"""Detect digital recapture of our own TTS; does not mute incoming audio."""
from collections import deque
import math
import threading
import time
import numpy as np
from scipy.signal import correlate, resample_poly


class PlaybackEchoGuard:
    def __init__(self):
        self.references = deque(maxlen=12)
        self.lock = threading.Lock()

    @staticmethod
    def _compact(audio, rate):
        divisor = math.gcd(rate, 2000)
        return resample_poly(audio, 2000//divisor, rate//divisor).astype(np.float32)

    def remember(self, audio, rate):
        compact = self._compact(audio, rate)
        if len(compact) < 1600:
            return
        with self.lock:
            self.references.append((time.perf_counter(), compact))

    def matches(self, audio, rate):
        signal = self._compact(audio, rate).astype(np.float64)
        if len(signal) < 1600:
            return False
        now = time.perf_counter()
        with self.lock:
            references = [ref for timestamp, ref in self.references if now-timestamp < 45]
        for reference in references:
            width = min(4000, len(reference), len(signal))
            sums = np.concatenate(([0], np.cumsum(signal)))
            squares = np.concatenate(([0], np.cumsum(signal*signal)))
            variance = np.maximum(0, squares[width:]-squares[:-width]
                                  -(sums[width:]-sums[:-width])**2/width)
            for offset in sorted({0, max(0, (len(reference)-width)//2), len(reference)-width}):
                template = reference[offset:offset+width].astype(np.float64)
                template -= template.mean()
                norm = np.linalg.norm(template)
                if norm < 1e-4:
                    continue
                numerator = correlate(signal, template, mode='valid', method='fft')
                denominator = norm*np.sqrt(variance)
                score = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                  where=denominator > 1e-8)
                if np.max(np.abs(score)) > .8:
                    return True
        return False
