"""Wall-clock replay of recorded speech, with simulated playback (no live audio I/O)."""
import argparse
import json
from pathlib import Path
import queue
import threading
import time
import wave
import numpy as np
import translator_v3_local as app
from speech_segments import SileroVAD, SpeechSegmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--mode', choices=['before', 'after'], default='after')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    app.CFG.adaptive_timing = args.mode == 'after'
    with wave.open(str(args.input), 'rb') as wav:
        if wav.getsampwidth() != 2:
            raise ValueError('16-bit PCM WAV required')
        audio = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32)/32768
        audio = audio.reshape(-1, wav.getnchannels()).mean(axis=1)
        audio = app.resample(audio, wav.getframerate(), 16000)
    if args.repeat < 1:
        raise ValueError('--repeat must be positive')
    audio = np.tile(audio, args.repeat)
    source_seconds = len(audio)/16000
    audio = np.concatenate((audio, np.zeros(16000, np.float32)))
    audio = np.pad(audio, (0, (-len(audio)) % 512))
    vad = SileroVAD(app.CFG.vad_model)
    segmenter = (SpeechSegmenter(vad, silence_ms=384, max_seconds=6.5, hard_seconds=8,
                                first_seconds=8, overlap_ms=0)
                 if args.mode == 'before' else SpeechSegmenter(vad))
    pipeline = app.Pipeline(None)
    playing = {'first': None, 'until': 0.0, 'audio_seconds': 0.0, 'chunks': 0}
    stopping = threading.Event()
    def consume():
        while not stopping.is_set() or not app.PLAY_QUEUE.empty():
            try:
                item = app.PLAY_QUEUE.get(timeout=.05)
            except queue.Empty:
                continue
            now = time.perf_counter()
            if playing['first'] is None:
                playing['first'] = now
            duration = len(item.samples)/item.sample_rate
            playing['until'] = max(now, playing['until']) + duration
            playing['audio_seconds'] += duration
            playing['chunks'] += 1
            app.PLAY_QUEUE.task_done()
    consumer = threading.Thread(target=consume, daemon=True)
    try:
        pipeline.start()
        startup = time.perf_counter()-pipeline.startup_started
        consumer.start()
        started = time.perf_counter()
        emitted = []
        for offset in range(0, len(audio), 512):
            deadline = started+(offset+512)/16000
            remaining = deadline-time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            segment = segmenter.feed(audio[offset:offset+512], started+offset/16000)
            if segment:
                emitted.append(round(time.perf_counter()-started, 3))
                app.AUDIO_QUEUE.put(segment)
        # Joins have a deadline so a failed worker cannot hang the benchmark.
        drain_deadline = time.perf_counter()+60
        queues = [app.AUDIO_QUEUE, pipeline.translation_jobs, pipeline.speech_jobs, app.PLAY_QUEUE]
        while any(q.unfinished_tasks for q in queues):
            if time.perf_counter() > drain_deadline:
                raise TimeoutError('Replay workers did not drain in 60 seconds')
            time.sleep(.05)
        result = {'mode': args.mode, 'repeat': args.repeat, 'startup_seconds': round(startup, 3),
                  'source_seconds': round(source_seconds, 3), 'segment_emissions': emitted,
                  'first_audio_seconds': round(playing['first']-started, 3) if playing['first'] else None,
                  'simulated_speech_after_source_stop_seconds': round(max(0, playing['until']-started-source_seconds), 3),
                  'generated_audio_seconds': round(playing['audio_seconds'], 3),
                  'chunks': playing['chunks'], 'content_policy': 'preserve',
                  'note': 'Real models and wall-clock capture replay; playback clock simulated, no headphone measurement'}
        suffix = f'-{args.repeat}' if args.repeat > 1 else ''
        report = app.ROOT/'samples'/f'sync-{args.mode}{suffix}.json'
        report.parent.mkdir(exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        stopping.set()
        app.RUNNING.clear()
        if consumer.is_alive():
            consumer.join(timeout=1)
        pipeline.stop()


if __name__ == '__main__':
    main()
