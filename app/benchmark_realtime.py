"""Offline VAD and first-phrase benchmark. Does not capture or play live audio."""
import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
from speech_segments import SileroVAD, SpeechSegmenter
from speech_text import speech_chunks
from translator_v3_local import ROOT, resample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True, help='16-bit PCM WAV speech fixture')
    parser.add_argument('--tts', action='store_true')
    args = parser.parse_args()
    with wave.open(str(args.input), 'rb') as wav:
        if wav.getsampwidth() != 2:
            raise ValueError('16-bit PCM WAV required')
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32)/32768
        audio = audio.reshape(-1, wav.getnchannels()).mean(axis=1)
        audio = resample(audio, wav.getframerate(), 16000)
    detector = SileroVAD(ROOT / 'models' / 'vad' / 'silero_vad.onnx')
    report = {'vad': {}, 'tts': {}}
    noise = np.random.default_rng(42).normal(0, .02, 16000*5).astype(np.float32)
    tone = (.05*np.sin(2*np.pi*180*np.arange(16000*5)/16000)).astype(np.float32)
    for name, fixture in [('silence', np.zeros(80000, np.float32)),
                          ('noise', noise), ('tone', tone), ('speech', audio)]:
        segmenter = SpeechSegmenter(detector)
        fixture = np.concatenate((np.zeros(16000, np.float32), fixture,
                                  np.zeros(16000, np.float32)))
        fixture = np.pad(fixture, (0, (-len(fixture)) % 512))
        segments, timings = [], []
        for index in range(0, len(fixture), 512):
            started = time.perf_counter()
            segment = segmenter.feed(fixture[index:index+512], index/16000)
            timings.append((time.perf_counter()-started)*1000)
            if segment:
                segments.append({'start': round(segment.started_at, 3),
                                 'end': round(segment.ended_at, 3),
                                 'emitted': round((index+512)/16000, 3),
                                 'endpoint_wait_ms': round(((index+512)/16000-segment.ended_at)*1000),
                                 'reason': segment.reason})
        report['vad'][name] = {'segments': segments,
                               'frame_p50_ms': round(float(np.median(timings)), 3),
                               'frame_p95_ms': round(float(np.percentile(timings, 95)), 3)}
    if args.tts:
        from local_voice import SupertonicVoice
        voice = SupertonicVoice()
        try:
            text = ('Bu ilk cümle doğal bir duraklama ile tamamlanıyor. '
                    'İkinci cümle önemli ayrıntıları açıklıyor ve konuşmanın bağlamını koruyor. '
                    'Son cümlede ise bütün ayrıntılar birlikte değerlendirilerek bir sonuca varılıyor.')
            chunks = speech_chunks(text)
            # Warm both input shapes to avoid reporting only shader compilation.
            voice.synthesize(text)
            voice.synthesize(chunks[0])
            full_times, first_times = [], []
            for _ in range(3):
                started = time.perf_counter()
                voice.synthesize(text)
                full_times.append(time.perf_counter()-started)
                started = time.perf_counter()
                stream = voice.stream(text)
                next(stream)
                first_times.append(time.perf_counter()-started)
                stream.close()
            report['tts'] = {'whole_text_first_audio_s': full_times,
                             'phrase_stream_first_audio_s': first_times,
                             'chunks': chunks, 'provider': voice.config['provider'],
                             'note': 'Synthesis only, not source-to-headphone latency'}
        finally:
            voice.stop()
    report_path = ROOT / 'samples' / 'realtime-benchmark.json'
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
