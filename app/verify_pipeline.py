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
        item = app.PLAY_QUEUE.get(timeout=30)
        assert item.label == 'EN->TR', item.label
        assert len(item.samples) > item.sample_rate*.1
        print(f'PASS: VAD -> STT -> MT -> TTS -> output queue; {len(item.samples)/item.sample_rate:.2f}s audio', flush=True)
        print('Headphone playback and live capture latency not tested.', flush=True)
    except queue.Empty:
        raise AssertionError('No English-to-Turkish audio within 30 seconds')
    finally:
        app.RUNNING.clear()
        pipeline.stop()


if __name__ == '__main__':
    main()
