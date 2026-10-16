"""Regression tests for audio processing and playback."""
import queue
import unittest
from types import SimpleNamespace
import numpy as np
from local_voice import clean_audio
from translator_v3_local import Player, resample


class AudioTests(unittest.TestCase):
    def player(self):
        player = Player.__new__(Player)
        player.blocks = queue.Queue()
        player.pending = np.empty(0, np.float32)
        player.underflows = 0
        return player

    def test_idle_output_is_exact_silence(self):
        player = self.player()
        output = np.ones((512, 1), np.float32)
        player.callback(output, 512, None, SimpleNamespace(output_underflow=False))
        self.assertFalse(output.any())

    def test_callback_keeps_all_samples_across_boundaries(self):
        player = self.player()
        player.blocks.put(np.array([1, 2, 3], np.float32))
        player.blocks.put(np.array([4, 5, 6], np.float32))
        output = np.zeros((4, 1), np.float32)
        player.callback(output, 4, None, SimpleNamespace(output_underflow=False))
        np.testing.assert_array_equal(output[:, 0], [1, 2, 3, 4])
        player.callback(output, 4, None, SimpleNamespace(output_underflow=False))
        np.testing.assert_array_equal(output[:, 0], [5, 6, 0, 0])

    def test_cleanup_removes_dc_and_softens_edges(self):
        output = clean_audio(np.full(44100, 0.1, np.float32), 44100)
        self.assertLess(float(np.max(abs(output[4410:]))), 1e-5)
        self.assertEqual(output[0], 0)
        self.assertEqual(output[-1], 0)

    def test_cleanup_does_not_amplify_quiet_noise(self):
        rng = np.random.default_rng(5)
        output = clean_audio(rng.normal(0, 1e-5, 44100), 44100)
        self.assertLess(np.sqrt(np.mean(output**2)), 2e-5)

    def test_resampler_filters_above_target_nyquist(self):
        audio = np.sin(2*np.pi*12000*np.arange(48000)/48000).astype(np.float32)
        output = resample(audio, 48000, 16000)
        self.assertEqual(len(output), 16000)
        self.assertLess(np.sqrt(np.mean(output[100:-100]**2)), 0.005)


if __name__ == '__main__':
    unittest.main()
