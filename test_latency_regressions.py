import unittest
from unittest.mock import Mock
import numpy as np
import translator_v3_local as app


class RegressionTests(unittest.TestCase):
    def test_language_confidence_is_not_promoted_to_one(self):
        model = app.WhisperVulkan()
        response = Mock()
        response.json.return_value = {
            'text': 'Hello', 'detected_language': 'english',
            'detected_language_probability': 0.34,
        }
        model.session = Mock()
        model.session.post.return_value = response
        _, language, probability = model.transcribe(np.ones(16000, dtype=np.float32)*0.1)
        self.assertEqual(language, 'en')
        self.assertEqual(probability, 0.34)

    def test_silence_does_not_call_recognizer(self):
        model = app.WhisperVulkan()
        model.session = Mock()
        self.assertEqual(model.transcribe(np.zeros(16000)), ('', '', 0.0))
        model.session.post.assert_not_called()

    def test_pause_after_phrase_flushes_without_periodic_cutting(self):
        while not app.AUDIO_QUEUE.empty():
            app.AUDIO_QUEUE.get_nowait()
        capture = app.Capture(None)
        speech = np.full((960, 1), 0.05, np.float32)
        for _ in range(40):
            capture.callback(speech, 960, None, None)
        self.assertTrue(app.AUDIO_QUEUE.empty())
        for _ in range(4):
            capture.callback(np.zeros_like(speech), 960, None, None)
        self.assertEqual(app.AUDIO_QUEUE.qsize(), 1)
        self.assertAlmostEqual(len(app.AUDIO_QUEUE.get().samples)/16000, 2.64)

    def test_missing_argos_separator_does_not_repeat_context(self):
        translator = app.LocalTranslator.__new__(app.LocalTranslator)
        translator.previous_source = 'Previous sentence.'
        translator.translation = Mock()
        translator.translation.translate.side_effect = ['Old and new together', 'Only new']
        self.assertEqual(translator._translate_with_argos_context('New sentence.'), 'Only new')


if __name__ == '__main__':
    unittest.main()
