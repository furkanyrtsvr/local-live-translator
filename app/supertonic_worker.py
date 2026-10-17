"""Isolated local ONNX worker; never downloads models or contacts a service."""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from realtime_policy import bounded_speed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'vendor' / 'supertonic' / 'py'))
from helper import TextToSpeech, load_cfgs, load_text_processor, load_voice_style


class Engine:
    def __init__(self, provider='cpu', voice='M1', steps=8, speed=1.12):
        model = ROOT / 'models' / 'supertonic-3'
        directory = str(model / 'onnx')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3
        providers = ['CPUExecutionProvider']
        if provider == 'dml':
            if 'DmlExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('DirectML provider unavailable')
            opts.enable_mem_pattern = False
            providers.insert(0, 'DmlExecutionProvider')
        sessions = [ort.InferenceSession(str(model / 'onnx' / (name + '.onnx')),
                    sess_options=opts, providers=providers) for name in
                    ('duration_predictor', 'text_encoder', 'vector_estimator', 'vocoder')]
        self.model = TextToSpeech(load_cfgs(directory), load_text_processor(directory), *sessions)
        self.style = load_voice_style([str(model / 'voice_styles' / (voice + '.json'))])
        self.steps, self.speed = steps, speed

    def generate(self, text, target_seconds=None, max_speed=1.55):
        speed = self.speed
        if target_seconds is not None:
            ids, mask = self.model.text_processor([text], ['tr'])
            duration, *_ = self.model.dp_ort.run(None, {
                'text_ids': ids, 'style_dp': self.style.dp, 'text_mask': mask})
            speed = bounded_speed(float(duration[0]), target_seconds, self.speed, max_speed)
        self.last_speed = speed
        wav, duration = self.model(text, 'tr', self.style, self.steps, speed,
                                   silence_duration=0.12)
        rate = self.model.sample_rate
        audio = np.asarray(wav[0, :int(float(duration[0]) * rate)], dtype=np.float32)
        if not np.isfinite(audio).all():
            raise ValueError('Non-finite model output')
        return audio, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=['cpu', 'dml'], default='cpu')
    parser.add_argument('--voice', default='M1')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--speed', type=float, default=1.12)
    args = parser.parse_args()
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    engine = Engine(args.provider, args.voice, args.steps, args.speed)
    engine.generate('Hazır.')
    print(json.dumps({'ready': True, 'provider': args.provider}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            started = time.perf_counter()
            audio, rate = engine.generate(request['text'], request.get('target_seconds'),
                                          request.get('max_speed', 1.55))
            result = {'audio': base64.b64encode(audio.astype('<f4').tobytes()).decode('ascii'),
                      'rate': rate, 'seconds': time.perf_counter() - started,
                      'speed': engine.last_speed}
        except Exception as exc:
            result = {'error': str(exc)}
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
