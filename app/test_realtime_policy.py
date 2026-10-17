import unittest
from realtime_policy import bounded_speed, phrase_budget, TranscriptOverlap


class TimingTests(unittest.TestCase):
    def test_duration_budget_speeds_up_without_exceeding_limit(self):
        self.assertEqual(bounded_speed(6, 4, 1.12, 1.55), 1.5)
        self.assertEqual(bounded_speed(12, 4, 1.12, 1.55), 1.55)
        self.assertEqual(bounded_speed(2, 4, 1.12, 1.55), 1.12)

    def test_phrase_budgets_preserve_total_duration_and_reduce_drift(self):
        budgets = [phrase_budget(4, words, 10) for words in [3, 7]]
        self.assertAlmostEqual(sum(budgets), 4)
        self.assertLess(phrase_budget(4, 10, 10, 5), 4)
        self.assertGreaterEqual(phrase_budget(4, 10, 10, 100), 3.2)

    def test_overlap_removes_repeated_prefix_not_new_words(self):
        overlap = TranscriptOverlap()
        overlap.apply('We are launching a new oatmeal', 'en', False)
        self.assertEqual(overlap.apply('new oatmeal raisin cookie.', 'en', True), 'raisin cookie.')
        self.assertEqual(overlap.apply('cookie recipes are here.', 'en', False), 'cookie recipes are here.')

    def test_overlap_does_not_suppress_language_change(self):
        overlap = TranscriptOverlap()
        overlap.apply('Merhaba.', 'tr', False)
        self.assertEqual(overlap.apply('Merhaba.', 'en', True), 'Merhaba.')


if __name__ == '__main__':
    unittest.main()
