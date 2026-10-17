"""Compare actual local translation backends on a fixed, inspectable sentence set."""
import argparse
import json
from pathlib import Path
import time
import wave
import numpy as np
import translator_v3_local as app

CASES = [
    ("Dive in. Let's start with the basics. To follow along, just go to Gemini.google.com and sign in with your Google account.",
     "Hadi başlayalım. Önce temel konuları ele alalım. Takip etmek için Gemini.google.com adresine gidip Google hesabınızla oturum açın."),
    ("Let's try it out. I run a small business. We're launching a new oatmeal raisin cookie.",
     "Deneyelim. Küçük bir işletmem var. Yeni bir yulaflı kuru üzümlü kurabiyeyi piyasaya çıkarıyoruz."),
    ("One of the biggest things to know is you don't need the perfect prompt.",
     "Bilmeniz gereken en önemli şeylerden biri, mükemmel bir isteme ihtiyacınız olmadığıdır."),
    ("I can make it shorter, change the tone, or adjust the format.",
     "Bunu kısaltabilir, üslubunu değiştirebilir veya biçimini düzenleyebilirim."),
    ("I can edit this document directly.", "Bu belgeyi doğrudan düzenleyebilirim."),
    ("We only use organic ingredients.", "Yalnızca organik malzemeler kullanıyoruz."),
    ("We should not jump to conclusions before seeing the results.",
     "Sonuçları görmeden aceleyle hükme varmamalıyız."),
    ("She was exhausted, but she still had to finish her report.",
     "Çok yorgundu, ama yine de raporunu bitirmesi gerekiyordu."),
    ("The company launched a new product.", "Şirket yeni bir ürün piyasaya sürdü."),
    ("Because it was easy to use, it became very popular.",
     "Kullanımı kolay olduğu için çok popüler oldu."),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', nargs='+', choices=['legacy', 'quality'], default=['legacy', 'quality'])
    parser.add_argument('--audio', type=Path, help='Optional 16-bit PCM English fixture to compare ASR')
    args = parser.parse_args()
    path = app.ROOT / 'samples' / 'translation-comparison.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for profile in args.profiles:
        app.apply_profile(profile)
        translator = app.LocalTranslator()
        try:
            # Warm up; do not score model startup/shader compilation as normal latency.
            translator.translate('The system is ready.')
            rows = []
            for source, reference in CASES:
                translator.history.clear()
                translator.previous_source = ''
                started = time.perf_counter()
                output = translator.translate(source)
                row = dict(source=source, reference=reference, output=output,
                           seconds=round(time.perf_counter()-started, 3))
                rows.append(row)
                print(json.dumps({'profile': profile, **row}, ensure_ascii=False), flush=True)
            report[profile] = rows
            if args.audio:
                with wave.open(str(args.audio), 'rb') as wav:
                    if wav.getsampwidth() != 2:
                        raise ValueError('16-bit PCM WAV required')
                    audio = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32)/32768
                    audio = audio.reshape(-1, wav.getnchannels()).mean(axis=1)
                    audio = app.resample(audio, wav.getframerate(), 16000)
                whisper = app.WhisperVulkan()
                try:
                    whisper.start()
                    started = time.perf_counter()
                    text, language, confidence = whisper.transcribe(audio)
                    print(json.dumps({'profile': profile, 'asr': text, 'language': language,
                                      'confidence': confidence, 'seconds': round(time.perf_counter()-started, 3)},
                                     ensure_ascii=False), flush=True)
                    report[profile + '_asr'] = text
                finally:
                    whisper.stop()
        finally:
            translator.stop()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
