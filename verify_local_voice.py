"""Manual smoke test: real models, no microphone capture or headphone playback."""
import time
import wave
from pathlib import Path
import numpy as np
import translator_v3_local as app


def main():
    voice = whisper = translator = None
    try:
        voice = app.LocalTTS()
        whisper = app.WhisperVulkan()
        whisper.start()
        translator = app.LocalTranslator()
        for source in ['The system is ready.',
                       'She was exhausted, but she still had to finish her report.',
                       'We should not jump to conclusions before seeing the results.']:
            started = time.perf_counter()
            target = translator.translate(source)
            mt_seconds = time.perf_counter() - started
            started = time.perf_counter()
            audio, rate = voice.synthesize(target)
            tts_seconds = time.perf_counter()-started
            transcript, language, confidence = whisper.transcribe(app.resample(audio, rate, 16000))
            print({'source': source, 'translation': target, 'read_back': transcript,
                   'language': language, 'confidence': confidence,
                   'translation_s': round(mt_seconds, 3), 'tts_s': round(tts_seconds, 3),
                   'audio_s': round(len(audio)/rate, 3)}, flush=True)
        audio, rate = voice.synthesize('Bence burada önemli olan yalnızca hız değil, konuşmanın doğal akması. Peki, bunu nasıl başaracağız? Önce sesi iyileştireceğiz, sonra çeviriyi birlikte değerlendireceğiz.')
        sample_dir = Path(__file__).parent / 'samples'
        sample_dir.mkdir(exist_ok=True)
        with wave.open(str(sample_dir / 'yeni_ses_temiz.wav'), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes((np.clip(audio, -1, 1)*32767).astype('<i2').tobytes())
    finally:
        for component in (voice, whisper, translator):
            if component is not None:
                component.stop()


if __name__ == '__main__':
    main()
