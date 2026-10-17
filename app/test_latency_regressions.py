"""Regression tests for capture, recognition and translation."""
import unittest
from unittest.mock import Mock
import numpy as np
import translator_v3_local as app


class RegressionTests(unittest.TestCase):
    def test_language_confidence_is_not_promoted_to_one(self):
        model = app.WhisperVulkan.__new__(app.WhisperVulkan)
        model.port = 8178
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
        model = app.WhisperVulkan.__new__(app.WhisperVulkan)
        model.session = Mock()
        self.assertEqual(model.transcribe(np.zeros(16000)), ('', '', 0.0))
        model.session.post.assert_not_called()

    def test_capture_callback_only_enqueues_raw_audio(self):
        capture = app.Capture(None)
        speech = np.full((512, 1), 0.05, np.float32)
        capture.callback(speech, 512, None, None)
        frame, started, drops, errors = capture.raw_frames.get_nowait()
        np.testing.assert_array_equal(frame, speech[:, 0])
        self.assertGreater(started, 0)
        self.assertEqual((drops, errors), (0, 0))

    def test_capture_overload_does_not_block_callback(self):
        capture = app.Capture(None)
        speech = np.full((512, 1), 0.05, np.float32)
        for _ in range(100):
            capture.callback(speech, 512, None, None)
        self.assertEqual(capture.raw_frames.qsize(), 64)
        self.assertEqual(capture.dropped_frames, 36)

    def test_missing_argos_separator_does_not_repeat_context(self):
        translator = app.LocalTranslator.__new__(app.LocalTranslator)
        translator.previous_source = 'Previous sentence.'
        translator.translation = Mock()
        translator.translation.translate.side_effect = ['Old and new together', 'Only new']
        self.assertEqual(translator._translate_with_argos_context('New sentence.'), 'Only new')

    def test_recognition_hands_off_without_running_translation(self):
        pipeline = app.Pipeline.__new__(app.Pipeline)
        pipeline.whisper = Mock()
        pipeline.echo_guard = Mock()
        pipeline.overlap_text = app.TranscriptOverlap()
        pipeline.whisper.transcribe.return_value = ('Hello.', 'en', .99)
        pipeline.translator = Mock()
        pipeline.translation_jobs = app.queue.Queue()
        segment = app.AudioSegment(np.zeros(16000, np.float32), 1, 2)
        pipeline.process(segment)
        self.assertEqual(pipeline.translation_jobs.get(), (segment, 'Hello.', 'en', .99))
        pipeline.translator.translate.assert_not_called()

    def test_turkish_pass_through_does_not_call_translation_or_tts(self):
        pipeline = app.Pipeline.__new__(app.Pipeline)
        pipeline.translator = Mock()
        pipeline.tts = Mock()
        pipeline.submit_speech = Mock()
        segment = app.AudioSegment(np.ones(16000, np.float32)*.1, 1, 2)
        pipeline.translate_result(segment, 'Merhaba.', 'tr', .99)
        item = pipeline.submit_speech.call_args.args[0]
        np.testing.assert_array_equal(item.samples, segment.samples)
        self.assertEqual(item.label, 'TR dogrudan')
        pipeline.translator.translate.assert_not_called()
        pipeline.tts.stream.assert_not_called()


if __name__ == '__main__':
    unittest.main()
