import unittest
import numpy as np
from speech_segments import SpeechSegmenter
from speech_text import speech_chunks


class FakeVAD:
    def reset(self):
        pass

    def __call__(self, frame):
        return float(frame[0])


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.segmenter = SpeechSegmenter(FakeVAD())
        self.clock = 100.0

    def feed(self, probabilities):
        output = []
        for probability in probabilities:
            segment = self.segmenter.feed(np.full(512, probability, np.float32), self.clock)
            self.clock += .032
            if segment:
                output.append(segment)
        return output

    def test_noise_does_not_create_segments(self):
        self.assertEqual(self.feed([.1] * 400), [])
        self.assertLessEqual(len(self.segmenter.preroll), 6)

    def test_weak_syllables_survive_hysteresis_and_short_pause(self):
        self.assertEqual(self.feed([.9]*30 + [.4]*8 + [.1]*4 + [.9]*30), [])
        segments = self.feed([.1]*8)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].reason, 'pause')
        self.assertGreater(segments[0].ended_at, 102)
        self.assertAlmostEqual(len(segments[0].samples)/16000, (30+8+4+30+3)*.032)

    def test_short_impulse_is_not_speech(self):
        self.assertEqual(self.feed([.9]*3 + [.1]*20), [])

    def test_long_phrase_uses_short_pause(self):
        self.assertEqual(self.feed([.9]*204), [])
        result = self.feed([.1]*3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].reason, 'long_pause')

    def test_hard_limit_is_bounded_and_does_not_duplicate_audio(self):
        results = self.feed([.9]*500)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(s.reason == 'limit' for s in results))
        self.assertEqual(sum(len(s.samples) for s in results), 500*512)
        self.assertAlmostEqual(results[1].started_at, results[0].started_at+8)

    def test_reset_does_not_join_audio_across_capture_gap(self):
        self.feed([.9]*15)
        self.segmenter.reset()
        self.assertEqual(self.feed([.1]*20), [])

    def test_short_sentences_stay_together(self):
        self.assertEqual(speech_chunks('Merhaba. Nasılsın?'), ['Merhaba. Nasılsın?'])

    def test_long_text_yields_first_sentence_and_preserves_every_word(self):
        text = ('Bu ilk cümle doğal bir duraklama ile tamamlanıyor. '
                'İkinci cümle önemli ayrıntıları açıklıyor ve konuşmanın bağlamını koruyor. '
                'Son cümlede ise bütün ayrıntılar birlikte değerlendirilerek bir sonuca varılıyor.')
        chunks = speech_chunks(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(' '.join(chunks), text)
        self.assertEqual(chunks[0], text.split('. ')[0]+'.')

    def test_abbreviations_are_not_sentence_boundaries(self):
        text = 'Bu konu hakkında uzman bir hekim olan Dr. Deniz uzun bir açıklama yaptı. ' + 'Ayrıntılar burada. '*15
        self.assertTrue(speech_chunks(text)[0].endswith('yaptı.'))


if __name__ == '__main__':
    unittest.main()
