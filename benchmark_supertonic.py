"""Generate reproducible listening samples and measure warmed synthesis only."""
import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
from supertonic_worker import Engine

parser = argparse.ArgumentParser()
parser.add_argument('--provider', default='cpu')
parser.add_argument('--voices', nargs='+', default=['M1', 'F1'])
args = parser.parse_args()
out = Path(__file__).resolve().parent / 'samples'
out.mkdir(exist_ok=True)
text = 'Bence burada önemli olan yalnızca hız değil, konuşmanın doğal akması. Peki, bunu nasıl başaracağız? Önce sesi iyileştireceğiz, sonra çeviriyi birlikte değerlendireceğiz.'
rows = []
for voice in args.voices:
    started = time.perf_counter()
    engine = Engine(args.provider, voice)
    engine.generate('Sistem hazır.')
    print('Ready', args.provider, voice, round(time.perf_counter()-started, 2), flush=True)
    for i, sentence in enumerate(['Bu sistem İngilizce konuşmayı Türkçeye çeviriyor.', text]):
        np.random.seed(42)
        started = time.perf_counter()
        audio, rate = engine.generate(sentence)
        elapsed = time.perf_counter()-started
        file = out / f'supertonic_{args.provider}_{voice}_{i}.wav'
        with wave.open(str(file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes((np.clip(audio, -1, 1)*32767).astype('<i2').tobytes())
        row = dict(provider=args.provider, voice=voice, text=sentence, seconds=elapsed,
                   audio_seconds=len(audio)/rate, peak=float(abs(audio).max()),
                   rtf=elapsed/(len(audio)/rate))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
print('DONE', flush=True)
