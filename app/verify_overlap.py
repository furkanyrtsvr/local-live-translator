"""Check forced streaming boundaries against a whole-file Whisper transcript."""
import argparse
import re
import time
import wave
from pathlib import Path
import numpy as np
import translator_v3_local as app
from realtime_policy import TranscriptOverlap
from speech_segments import SpeechSegmenter


class ContinuousSpeech:
    def reset(self):
        pass

    def __call__(self, frame):
        return .99


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
    audio = np.pad(audio, (0, (-len(audio)) % 512))
    segmenter = SpeechSegmenter(ContinuousSpeech())
    overlap = TranscriptOverlap()
    whisper = app.WhisperVulkan()
    pieces = []
    try:
        whisper.start()
        whole = whisper.transcribe(audio)[0]
        def recognise(samples, overlapping):
            text, language, _ = whisper.transcribe(samples)
            result = overlap.apply(text, language, overlapping)
            pieces.append(result)
            print(result, flush=True)
        started = time.perf_counter()
        for offset in range(0, len(audio), 512):
            segment = segmenter.feed(audio[offset:offset+512], started+offset/16000)
            if segment:
                recognise(segment.samples, segment.overlap_samples > 0)
        if segmenter.frames:
            recognise(np.concatenate(segmenter.frames), segmenter.overlap_samples > 0)
        normalise = lambda s: re.findall(r'\w+', s.casefold())
        matches = normalise(whole) == normalise(' '.join(pieces))
        print(f'Whole transcript: {whole}', flush=True)
        print(f'Forced-boundary word sequence matches whole-file decode: {matches}', flush=True)
        if not matches:
            raise AssertionError('Boundary transcription differs; inspect output, do not assume content preserved')
    finally:
        whisper.stop()


if __name__ == '__main__':
    main()
