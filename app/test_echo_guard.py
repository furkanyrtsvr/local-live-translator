import unittest
import numpy as np
from echo_guard import PlaybackEchoGuard


class EchoTests(unittest.TestCase):
    def test_matches_scaled_digital_playback_inside_longer_capture(self):
        rng = np.random.default_rng(14)
        voice = rng.normal(0, .1, 16000*3).astype(np.float32)
        guard = PlaybackEchoGuard()
        guard.remember(voice, 16000)
        captured = np.concatenate((np.zeros(16000), voice*.4, np.zeros(16000))).astype(np.float32)
        captured += rng.normal(0, .003, len(captured)).astype(np.float32)
        self.assertTrue(guard.matches(captured, 16000))

    def test_unrelated_audio_is_not_suppressed(self):
        rng = np.random.default_rng(15)
        guard = PlaybackEchoGuard()
        guard.remember(rng.normal(0, .1, 48000).astype(np.float32), 16000)
        self.assertFalse(guard.matches(rng.normal(0, .1, 48000).astype(np.float32), 16000))
        self.assertFalse(guard.matches(np.zeros(48000, np.float32), 16000))


if __name__ == '__main__':
    unittest.main()
