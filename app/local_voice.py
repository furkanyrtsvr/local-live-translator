"""Persistent, offline Supertonic subprocess (keeps DirectML dependencies isolated)."""
import base64
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
from speech_text import speech_chunks
from realtime_policy import phrase_budget

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent


def clean_audio(audio, rate):
    """Remove sub-bass/DC, prevent overload and soften boundaries; no peak boost."""
    audio = np.asarray(audio, dtype=np.float32).copy()
    if not len(audio):
        return audio
    if not np.isfinite(audio).all():
        raise ValueError('Ses verisi geçersiz')
    audio = sosfilt(butter(2, 45, 'highpass', fs=rate, output='sos'), audio).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.90:
        audio *= 0.90 / peak
    fade = min(int(rate * 0.004), len(audio)//2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return audio


class SupertonicVoice:
    def __init__(self):
        config_path = APP_DIR / 'voice_config.json'
        self.config = json.loads(config_path.read_text(encoding='utf-8'))
        python = ROOT / '.venv-supertonic' / 'Scripts' / 'python.exe'
        self.responses = queue.Queue()
        self.log = (ROOT / 'supertonic_worker.log').open('a', encoding='utf-8')
        try:
            self.process = subprocess.Popen(
                [str(python), '-u', str(APP_DIR / 'supertonic_worker.py'),
                 '--provider', self.config['provider'], '--voice', self.config['voice'],
                 '--steps', str(self.config['steps']), '--speed', str(self.config['speed'])],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding='utf-8', bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            if not self._response(120).get('ready'):
                raise RuntimeError('Supertonic başlatılamadı')
        except Exception:
            self.stop()
            raise
        print(f"Supertonic 3: HAZIR ({self.config['provider']}, {self.config['voice']}, tamamen yerel)")

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except Exception as exc:
            self.responses.put({'error': str(exc)})
        finally:
            self.responses.put({'error': 'Ses işlemi kapandı; supertonic_worker.log dosyasına bakın'})

    def _response(self, timeout):
        try:
            response = self.responses.get(timeout=timeout)
        except queue.Empty:
            self.stop()
            raise TimeoutError('Supertonic zaman aşımı')
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def synthesize(self, text, target_seconds=None):
        if not text.strip():
            return np.empty(0, np.float32), 44100
        self.process.stdin.write(json.dumps({'text': text, 'target_seconds': target_seconds,
                                             'max_speed': self.config.get('max_speed', 1.55)},
                                            ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        response = self._response(45)
        self.last_speed = response.get('speed', self.config['speed'])
        audio = np.frombuffer(base64.b64decode(response['audio']), dtype='<f4').copy()
        return clean_audio(audio, response['rate']), response['rate']

    def stream(self, text, source_seconds=None, backlog_seconds=0):
        # The first complete phrase can play while subsequent phrases synthesize.
        # This is sentence streaming, not sample streaming inside the neural model.
        phrases = speech_chunks(text, minimum=35, maximum=120)
        total_words = sum(len(phrase.split()) for phrase in phrases)
        for phrase in phrases:
            budget = (phrase_budget(source_seconds, len(phrase.split()), total_words, backlog_seconds)
                      if source_seconds is not None else None)
            yield self.synthesize(phrase, budget)

    def stop(self):
        process = getattr(self, 'process', None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if hasattr(self, 'log'):
            self.log.close()
