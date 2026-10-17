"""Real worker smoke test using an English WAV fixture; no live audio I/O."""
import argparse
import queue
import time
import wave
from pathlib import Path
import numpy as np
import translator_v3_local as app
from speech_segments import SpeechSegmenter, SileroVAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    with wave.open(str(args.input), 'rb') as wav:
        if wav.getsampwidth() != 2:
            raise ValueError('16-bit PCM WAV required')
        audio = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32)/32768
        audio = audio.reshape(-1, wav.getnchannels()).mean(axis=1)
        audio = app.resample(audio, wav.getframerate(), 16000)
    segmenter = SpeechSegmenter(SileroVAD(app.CFG.vad_model))
    audio = np.concatenate((audio, np.zeros(16000, np.float32)))
    audio = np.pad(audio, (0, (-len(audio)) % 512))
    pipeline = app.Pipeline(None)
    try:
        pipeline.start()
        started = time.perf_counter()
        for offset in range(0, len(audio), 512):
            segment = segmenter.feed(audio[offset:offset+512], started-len(audio)/16000+offset/16000)
            if segment:
                app.AUDIO_QUEUE.put(segment)
        items = []
        deadline = time.perf_counter()+45
        while True:
            try:
                item = app.PLAY_QUEUE.get(timeout=.1)
                items.append(item)
                app.PLAY_QUEUE.task_done()
            except queue.Empty:
                pass
            if not any(q.unfinished_tasks for q in
                       (app.AUDIO_QUEUE, pipeline.translation_jobs, pipeline.speech_jobs, app.PLAY_QUEUE)):
                break
            if time.perf_counter() > deadline:
                raise TimeoutError('Pipeline did not drain in 45 seconds')
        assert items and all(item.label == 'EN->TR' for item in items)
        assert all(len(item.samples) > item.sample_rate*.1 for item in items)
        print(f'PASS: all VAD -> STT -> MT -> TTS jobs drained; {len(items)} audio chunks', flush=True)
        print('Headphone playback and live capture latency not tested.', flush=True)
    except queue.Empty:
        raise AssertionError('No English-to-Turkish audio within 30 seconds')
    finally:
        app.RUNNING.clear()
        pipeline.stop()


if __name__ == '__main__':
    main()
