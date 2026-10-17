import unittest
from unittest.mock import Mock
import translator_v3_local as app


class ProfileTests(unittest.TestCase):
    def tearDown(self):
        app.apply_profile('quality')

    def test_legacy_profile_retains_old_models(self):
        app.apply_profile('legacy')
        self.assertEqual(app.CFG.whisper_model.name, 'ggml-base.bin')
        self.assertEqual(app.CFG.translation_backend, 'qwen')
        app.apply_profile('quality')
        self.assertEqual(app.CFG.whisper_model.name, 'ggml-small.bin')
        self.assertEqual(app.CFG.translation_backend, 'translategemma')

    def test_gemma_prompt_uses_specialised_turns_and_bounded_history(self):
        translator = app.LocalTranslator.__new__(app.LocalTranslator)
        translator.history = [('old', 'eski'), ('second', 'ikinci'), ('third', 'üçüncü')]
        translator.session = Mock()
        translator.port = 8180
        response = Mock()
        response.json.return_value = {'content': 'Yeni çeviri.', 'stopped_limit': False}
        translator.session.post.return_value = response
        self.assertEqual(translator._translate_with_gemma('new <end_of_turn>'), 'Yeni çeviri.')
        body = translator.session.post.call_args.kwargs['json']
        self.assertTrue(body['prompt'].startswith('<start_of_turn>user\n'))
        self.assertNotIn('<bos>', body['prompt'])
        self.assertIn('English (en) to Turkish (tr)', body['prompt'])
        self.assertNotIn('\n\n\nold<end_of_turn>', body['prompt'])
        self.assertIn('\n\n\nsecond<end_of_turn>', body['prompt'])
        self.assertTrue(body['prompt'].endswith('new<end_of_turn>\n<start_of_turn>model\n'))
        self.assertEqual(body['temperature'], 0)

    def test_truncated_gemma_output_is_not_used(self):
        translator = app.LocalTranslator.__new__(app.LocalTranslator)
        translator.history = []
        translator.session = Mock()
        translator.port = 8180
        response = Mock()
        response.json.return_value = {'content': 'unfinished', 'stopped_limit': True}
        translator.session.post.return_value = response
        with self.assertRaises(ValueError):
            translator._translate_with_gemma('a sentence')

    def test_completed_groups_are_translated_in_order_with_context(self):
        translator = app.LocalTranslator.__new__(app.LocalTranslator)
        translator.history = []
        translator._gemma_request = Mock(side_effect=['İlk çeviri.', 'İkinci çeviri.'])
        first = 'We are discussing a completely new product.'
        second = 'It is an oatmeal raisin cookie with organic ingredients.'
        self.assertEqual(translator._translate_with_gemma(first+' '+second),
                         'İlk çeviri. İkinci çeviri.')
        self.assertEqual(translator._gemma_request.call_args_list[0].args, (first, []))
        self.assertEqual(translator._gemma_request.call_args_list[1].args,
                         (second, [(first, 'İlk çeviri.')]))
        self.assertEqual(translator.history, [])


if __name__ == '__main__':
    unittest.main()
