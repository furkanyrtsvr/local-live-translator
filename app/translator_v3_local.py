"""Tamamen yerel, AMD Vulkan hızlandırmalı canlı İngilizce -> Türkçe çevirmen."""

from __future__ import annotations

import io
import argparse
import math
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
from argostranslate import translate as argos_translate
from scipy.signal import resample_poly
from local_voice import SupertonicVoice
from speech_segments import AudioSegment, SileroVAD, SpeechSegmenter
from speech_text import speech_chunks
from echo_guard import PlaybackEchoGuard

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent.parent
APP_VERSION = "0.3.0-rc.1"


@dataclass
class Config:
    sample_rate: int = 16000
    block_ms: int = 32
    language_probability: float = 0.52
    whisper_port: int = 8178
    whisper_threads: int = 8
    whisper_model: Path = ROOT / "models" / "whisper" / "ggml-small.bin"
    whisper_server: Path = ROOT / "vendor" / "whisper-vulkan" / "whisper-server.exe"
    llm_server: Path = ROOT / "vendor" / "llama-vulkan" / "llama-server.exe"
    llm_model: Path = ROOT / "models" / "llm" / "translategemma-4b-it.Q5_K_M.gguf"
    translation_backend: str = "translategemma"
    llm_port: int = 8180
    context_sentences: int = 3
    output_device: Optional[int] = None
    vad_model: Path = ROOT / "models" / "vad" / "silero_vad.onnx"


CFG = Config()


def apply_profile(profile):
    if profile == "legacy":
        CFG.whisper_model = ROOT / "models" / "whisper" / "ggml-base.bin"
        CFG.llm_model = ROOT / "models" / "llm" / "Qwen3-4B-Q4_K_M.gguf"
        CFG.translation_backend = "qwen"
    elif profile == "quality":
        CFG.whisper_model = ROOT / "models" / "whisper" / "ggml-small.bin"
        CFG.llm_model = ROOT / "models" / "llm" / "translategemma-4b-it.Q5_K_M.gguf"
        CFG.translation_backend = "translategemma"
    else:
        raise ValueError(f"Unknown profile: {profile}")


RUNNING = threading.Event()
RUNNING.set()
AUDIO_QUEUE: queue.Queue[AudioSegment] = queue.Queue(maxsize=8)
PLAY_QUEUE: queue.Queue[PlayItem] = queue.Queue(maxsize=8)


@dataclass
class PlayItem:
    samples: np.ndarray
    sample_rate: int
    label: str
    created_at: float
    source_started_at: float = 0.0


def available_port(preferred: int) -> int:
    """Eski bir programın model sunucusuna yanlışlıkla bağlanma."""
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
        except OSError:
            probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(samples) < 2:
        return samples.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)


def select_device(kind: str) -> Optional[int]:
    devices = sd.query_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    candidates: list[int] = []
    print(f"\n{'GIRIS (CABLE Output)' if kind == 'input' else 'CIKIS (kulaklik/hoparlor)'}")
    for index, device in enumerate(devices):
        if device[channel_key] <= 0:
            continue
        name = device["name"]
        low = name.lower()
        preferred = False
        if kind == "input":
            preferred = "cable output" in low and "point" not in low
        else:
            preferred = (any(word in low for word in
                             ("headphone", "kulak", "speaker", "hoparl", "realtek"))
                         and "cable" not in low)
        if preferred:
            candidates.append(index)
        print(f"  [{index:2}] {name}{'  <-- onerilen' if preferred else ''}")

    default = candidates[0] if candidates else None
    prompt = f"Numara (ENTER={default if default is not None else 'varsayilan'}): "
    answer = input(prompt).strip()
    return int(answer) if answer.isdigit() else default


class Capture:
    def __init__(self, device: Optional[int]):
        self.device = device
        self.raw_frames = queue.Queue(maxsize=64)
        self.stopping = threading.Event()
        self.dropped_frames = 0
        self.input_errors = 0
        self.thread = None
        self.stream = None

    def callback(self, data, frames, timing, status):
        if status:
            self.input_errors += 1
        mono = data[:, 0].copy() if data.ndim > 1 else data.copy().reshape(-1)
        # Callback only copies/enqueues. Model inference and array growth run outside it.
        started_at = time.perf_counter() - len(mono) / CFG.sample_rate
        try:
            self.raw_frames.put_nowait((mono, started_at, self.dropped_frames, self.input_errors))
        except queue.Full:
            self.dropped_frames += 1

    def loop(self):
        previous_gaps = (0, 0)
        while RUNNING.is_set() and not self.stopping.is_set():
            try:
                frame, started_at, drops, errors = self.raw_frames.get(timeout=.1)
            except queue.Empty:
                continue
            if (drops, errors) != previous_gaps:
                self.segmenter.reset()
                previous_gaps = (drops, errors)
                print(f"[GIRIS] Kesinti; VAD sifirlandi: kayip={drops}, cihaz={errors}")
            try:
                segment = self.segmenter.feed(frame, started_at)
                if segment is not None:
                    print(f"[VAD] {len(segment.samples)/CFG.sample_rate:.2f}s | {segment.reason}")
                    try:
                        AUDIO_QUEUE.put_nowait(segment)
                    except queue.Full:
                        AUDIO_QUEUE.get_nowait()
                        AUDIO_QUEUE.put_nowait(segment)
                        print("[UYARI] STT geride kaldi; en eski parca atlandi")
            except Exception as exc:
                print(f"[VAD HATASI] {exc}")
                self.segmenter.reset()

    def start(self):
        self.segmenter = SpeechSegmenter(SileroVAD(CFG.vad_model))
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=CFG.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(CFG.sample_rate * CFG.block_ms / 1000),
            latency="low",
            callback=self.callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=2)


class WhisperVulkan:
    def __init__(self):
        for path in (CFG.whisper_server, CFG.whisper_model):
            if not path.exists():
                raise FileNotFoundError(path)
        self.process: Optional[subprocess.Popen] = None
        self.server_log = None
        self.session = requests.Session()
        self.session.trust_env = False
        self.port = available_port(CFG.whisper_port)

    def start(self):
        command = [
            str(CFG.whisper_server), "-m", str(CFG.whisper_model),
            "-l", "auto", "-t", str(CFG.whisper_threads),
            "-bo", "1", "-bs", "1", "-nt", "-sns",
            "--host", "127.0.0.1", "--port", str(self.port),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.server_log = (ROOT / "whisper_server.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command, cwd=str(CFG.whisper_server.parent),
            stdout=self.server_log, stderr=self.server_log,
            creationflags=creationflags,
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Whisper Vulkan sunucusu baslatilamadi")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    print(f"Whisper/Vulkan: HAZIR ({CFG.whisper_model.name})")
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("Whisper Vulkan sunucusu 60 saniyede hazir olmadi; whisper_server.log")

    @staticmethod
    def wav_bytes(samples: np.ndarray) -> bytes:
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(CFG.sample_rate)
            wav.writeframes(pcm)
        return output.getvalue()

    def transcribe(self, samples: np.ndarray, language_hint: str = "auto") -> tuple[str, str, float]:
        if rms(samples) < 1e-5:
            return "", "", 0.0
        response = self.session.post(
            f"http://127.0.0.1:{self.port}/inference",
            files={"file": ("segment.wav", self.wav_bytes(samples), "audio/wav")},
            data={"response_format": "verbose_json", "language": language_hint,
                  "temperature": "0.0"},
            timeout=20,
        )
        response.raise_for_status()
        result = response.json()
        text = str(result.get("text", "")).strip()
        language = str(result.get("detected_language") or result.get("language", "")).lower()
        language = {"english": "en", "turkish": "tr"}.get(language, language)
        probability = float(result.get("detected_language_probability") or 0.0)
        probabilities = result.get("language_probabilities") or {}
        if not probability and isinstance(probabilities, dict) and language:
            probability = float(probabilities.get(language, 0.0))
        return text, language, probability

    def stop(self):
        self.session.close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.server_log:
            self.server_log.close()


class LocalTranslator:
    def __init__(self):
        languages = argos_translate.get_installed_languages()
        source = next((lang for lang in languages if lang.code == "en"), None)
        target = next((lang for lang in languages if lang.code == "tr"), None)
        if not source or not target:
            raise RuntimeError("Argos en->tr modeli kurulu degil")
        self.translation = source.get_translation(target)
        if not self.translation:
            raise RuntimeError("Argos en->tr cevirisi bulunamadi")
        self.translation.translate("Hello.")
        self.previous_source = ""
        self.history: list[tuple[str, str]] = []
        self.session = requests.Session()
        self.session.trust_env = False
        self.port = available_port(CFG.llm_port)
        self.process: Optional[subprocess.Popen] = None
        self.llm_ready = False
        self.server_log = None
        self._start_context_model()
        print(f"Ceviri: HAZIR ({CFG.translation_backend}/Vulkan, {CFG.llm_model.name})")

    def _start_context_model(self):
        if not CFG.llm_server.exists() or not CFG.llm_model.exists():
            if CFG.translation_backend == "translategemma":
                raise FileNotFoundError("Kalite modeli eksik. py -3.13 app/download_quality_models.py")
            print("[UYARI] Qwen bulunamadi; Argos baglam modu kullanilacak")
            return
        command = [
            str(CFG.llm_server), "-m", str(CFG.llm_model),
            "-ngl", "all", "-c", "2048", "-t", "8",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--reasoning", "off", "--no-webui",
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.server_log = (ROOT / "translation_server.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command, cwd=str(CFG.llm_server.parent),
            stdout=self.server_log, stderr=self.server_log,
            creationflags=creationflags,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.process.poll() is not None:
                break
            try:
                response = self.session.get(
                    f"http://127.0.0.1:{self.port}/health", timeout=0.4
                )
                if response.status_code == 200:
                    self.llm_ready = True
                    return
            except requests.RequestException:
                pass
            time.sleep(0.15)
        if CFG.translation_backend == "translategemma":
            self.stop()
            raise RuntimeError("TranslateGemma baslatilamadi; translation_server.log kaydina bakin")
        print("[UYARI] Qwen baslatilamadi; Argos kullanilacak")

    def translate(self, text: str) -> str:
        if self.llm_ready:
            try:
                current = (self._translate_with_gemma(text) if CFG.translation_backend == "translategemma"
                           else self._translate_with_context(text))
                if re.search(r"[\u3400-\u9fff]", current) and not re.search(r"[\u3400-\u9fff]", text):
                    raise ValueError("Ceviri Turkce disinda karakterler uretti")
                self.history.append((text, current))
                self.history = self.history[-CFG.context_sentences:]
                self.previous_source = text[-500:]
                return self._prepare_for_speech(current)
            except Exception as exc:
                print(f"[UYARI] Yerel ceviri hatasi, Argos'a gecildi: {exc}")

        current = self._translate_with_argos_context(text)
        self.history.append((text, current))
        self.history = self.history[-CFG.context_sentences:]
        return self._prepare_for_speech(current)

    @staticmethod
    def _gemma_turn(text):
        # Exact text contract embedded in TranslateGemma's tokenizer.chat_template.
        instruction = (
            "You are a professional English (en) to Turkish (tr) translator. Your goal is to "
            "accurately convey the meaning and nuances of the original English text while "
            "adhering to Turkish grammar, vocabulary, and cultural sensitivities.\n"
            "Produce only the Turkish translation, without any additional explanations or "
            "commentary. Please translate the following English text into Turkish:\n\n\n"
        )
        return "<start_of_turn>user\n" + instruction + text.strip() + "<end_of_turn>\n"

    def _translate_with_gemma(self, text):
        # Translating a whole multi-sentence audio chunk can lose details (e.g.
        # ingredients). Keep short adjacent sentences together and translate
        # completed groups in order, carrying bounded context between them.
        context = list(self.history[-2:])
        results = []
        for source in speech_chunks(text, minimum=30, maximum=1):
            target = self._gemma_request(source, context)
            results.append(target)
            context = (context + [(source, target)])[-2:]
        return " ".join(results)

    def _gemma_request(self, text, context):
        # /completion preserves TranslateGemma's specialised template; generic
        # chat endpoints may strip source_lang_code/target_lang_code content fields.
        escape = lambda s: re.sub(r"<(?:start_of_turn|end_of_turn|bos|eos)>", "", s)
        # llama.cpp automatically inserts the model's BOS token. Including it in
        # the text would create two BOS tokens (confirmed in server diagnostics).
        prompt = ""
        for source, target in context:
            prompt += self._gemma_turn(escape(source))
            prompt += "<start_of_turn>model\n" + escape(target) + "<end_of_turn>\n"
        prompt += self._gemma_turn(escape(text)) + "<start_of_turn>model\n"
        response = self.session.post(f"http://127.0.0.1:{self.port}/completion", json={
            "prompt": prompt, "temperature": 0, "n_predict": 256,
            "stop": ["<end_of_turn>", "<eos>"], "cache_prompt": True,
        }, timeout=30)
        response.raise_for_status()
        result = response.json()
        if result.get("stopped_limit"):
            raise ValueError("TranslateGemma token sinirinda kesildi")
        current = result.get("content", "").strip()
        if not current:
            raise ValueError("TranslateGemma bos sonuc verdi")
        return current

    def _translate_with_context(self, text: str) -> str:
        context = "\n".join(
            f"EN: {source}\nTR: {target}" for source, target in self.history
        ) or "(none)"
        body = {
            "model": "local",
            "temperature": 0.1,
            "max_tokens": 180,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate CURRENT from English into natural spoken Turkish. "
                        "Use CONTEXT only to preserve meaning, names, pronouns and terminology. "
                        "Make the Turkish flow naturally from prior sentences. Add appropriate "
                        "Turkish punctuation for expressive speech. Do not summarize, explain, "
                        "censor, or add information. Return only the current Turkish translation."
                    ),
                },
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nCURRENT:\n{text}"},
            ],
        }
        response = self.session.post(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            json=body, timeout=15,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Ceviri token sinirinda kesildi")
        result = choice["message"]["content"].strip()
        result = re.sub(r"^(TR|Türkçe|Çeviri)\s*:\s*", "", result, flags=re.IGNORECASE)
        result = result.strip(' "“”')
        if not result:
            raise ValueError("Qwen bos sonuc verdi")
        return result

    def _translate_with_argos_context(self, text: str) -> str:
        # Önceki İngilizce cümleyi bağlam olarak beraber çevir. Argos ayırıcı
        # çubukların sayısını değiştirebilse de koruduğu için son kısmı güvenle
        # alabiliriz. Böylece he/it/they gibi zamirler kör çevrilmez.
        if self.previous_source:
            combined = f"{self.previous_source} ||| {text}"
            translated = self.translation.translate(combined).strip()
            parts = [part.strip() for part in re.split(r"\|+", translated) if part.strip()]
            current = parts[-1] if len(parts) == 2 else self.translation.translate(text).strip()
        else:
            current = self.translation.translate(text).strip()
        self.previous_source = text[-500:]
        return current

    def stop(self):
        self.session.close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.server_log:
            self.server_log.close()

    @staticmethod
    def _prepare_for_speech(text: str) -> str:
        """Ses modeline verilecek Türkçe noktalama ve boşlukları temizle."""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        if text and text[-1] not in ".!?":
            text += "."
        return text


class LocalTTS(SupertonicVoice):
    pass


class Player:
    def __init__(self, device: Optional[int]):
        self.device = device
        device_info = sd.query_devices(device, "output")
        self.output_rate = int(device_info.get("default_samplerate", 44100))
        self.blocks = queue.Queue(maxsize=16)
        self.pending = np.empty(0, dtype=np.float32)
        self.underflows = 0
        self.stream = sd.OutputStream(
            device=device,
            samplerate=self.output_rate,
            channels=1,
            dtype="float32",
            latency=0.08,
            callback=self.callback,
        )
        self.stream.start()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def callback(self, outdata, frames, timing, status):
        # Always feed silence when idle. No disk, resampling or blocking in callback.
        outdata.fill(0)
        if status.output_underflow:
            self.underflows += 1
        offset = 0
        while offset < frames:
            if not len(self.pending):
                try:
                    self.pending = self.blocks.get_nowait()
                except queue.Empty:
                    break
            count = min(frames - offset, len(self.pending))
            outdata[offset:offset+count, 0] = self.pending[:count]
            self.pending = self.pending[count:]
            offset += count

    def loop(self):
        while RUNNING.is_set() or not PLAY_QUEUE.empty():
            try:
                item = PLAY_QUEUE.get(timeout=0.2)
            except queue.Empty:
                continue
            delay = (time.perf_counter() - item.created_at) * 1000
            total = (time.perf_counter() - item.source_started_at) * 1000 if item.source_started_at else 0
            print(f"[CAL] {item.label} | kuyruk={delay:.0f}ms | kaynak baslangicindan={total:.0f}ms")
            # Tek bir çıkış akışını sürekli açık tut: parçalar arasında cihazı
            # kapatıp açmanın oluşturduğu klik ve boşlukları önler.
            audio = resample(item.samples, item.sample_rate, self.output_rate)
            try:
                step = max(1, self.output_rate // 20)
                for offset in range(0, len(audio), step):
                    if not RUNNING.is_set():
                        break
                    while RUNNING.is_set():
                        try:
                            self.blocks.put(audio[offset:offset + step], timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                if RUNNING.is_set():
                    print(f"[CIKIS HATASI] {exc}")

    def stop(self):
        try:
            self.thread.join(timeout=1)
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class Pipeline:
    def __init__(self, output_device: Optional[int]):
        self.output_device = output_device
        self.echo_guard = PlaybackEchoGuard()
        self.whisper = WhisperVulkan()
        self.tts = LocalTTS()
        try:
            self.translator = LocalTranslator()
        except Exception:
            self.tts.stop()
            raise
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.translation_jobs = queue.Queue(maxsize=8)
        self.translation_thread = threading.Thread(target=self.translation_loop, daemon=True)
        self.speech_jobs = queue.Queue(maxsize=8)
        self.speech_thread = threading.Thread(target=self.speech_loop, daemon=True)

    def start(self):
        self.whisper.start()
        # İlk gerçek cümlede shader/model ısınması yaşanmasın.
        warm_audio, rate = self.tts.synthesize("Sistem hazır.")
        self.whisper.transcribe(resample(warm_audio, rate, CFG.sample_rate))
        self.translator.translate("The system is ready.")
        self.translator.history.clear()
        self.translator.previous_source = ""
        self.speech_thread.start()
        self.translation_thread.start()
        self.thread.start()

    def speech_loop(self):
        while RUNNING.is_set():
            try:
                job = self.speech_jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if isinstance(job, PlayItem):
                    self.enqueue(job)
                    continue
                translated, source_started = job
                wait_ms = (time.perf_counter() - source_started)*1000
                if wait_ms > 12000:
                    print(f"[GECIKME] Ses sirasi kaynagin {wait_ms/1000:.1f}s gerisinde")
                started = time.perf_counter()
                for audio, rate in self.tts.stream(translated):
                    if not RUNNING.is_set():
                        break
                    self.echo_guard.remember(audio, rate)
                    self.enqueue(PlayItem(audio, rate, "EN->TR", time.perf_counter(), source_started))
                print(f"[TTS] {(time.perf_counter()-started)*1000:.0f}ms")
            except Exception as exc:
                print(f"[TTS HATASI] {exc}")

    def submit_speech(self, job):
        while RUNNING.is_set():
            try:
                self.speech_jobs.put(job, timeout=0.1)
                return
            except queue.Full:
                continue

    def enqueue(self, item: PlayItem):
        while RUNNING.is_set():
            try:
                PLAY_QUEUE.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def loop(self):
        while RUNNING.is_set():
            try:
                segment = AUDIO_QUEUE.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process(segment)
            except Exception as exc:
                print(f"[PIPELINE HATASI] {exc}")

    def process(self, segment: AudioSegment):
        started = time.perf_counter()
        text, language, probability = self.whisper.transcribe(segment.samples)
        stt_ms = (time.perf_counter() - started) * 1000
        if not text:
            return
        print(f"[STT {stt_ms:.0f}ms] {language} %{probability*100:.0f}: {text}")
        if language == "tr" and self.echo_guard.matches(segment.samples, CFG.sample_rate):
            print("[SES DONUSU] Kendi uretilen sesimiz girise geri dondu; tekrar oynatma atlandi. "
                  "Windows cikisini/kulaklik yonlendirmesini kontrol edin.")
            return

        # Keep recognising while the previous result is being translated.
        while RUNNING.is_set():
            try:
                self.translation_jobs.put((segment, text, language, probability), timeout=.1)
                return
            except queue.Full:
                continue

    def translation_loop(self):
        while RUNNING.is_set():
            try:
                job = self.translation_jobs.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self.translate_result(*job)
            except Exception as exc:
                print(f"[CEVIRI HATASI] {exc}")

    def translate_result(self, segment, text, language, probability):

        if language == "tr" and probability >= CFG.language_probability:
            self.translator.history.clear()
            self.translator.previous_source = ""
            self.submit_speech(PlayItem(segment.samples, CFG.sample_rate, "TR dogrudan", time.perf_counter(), segment.started_at))
            return

        if language != "en" or probability < CFG.language_probability:
            print(f"[ATLANDI] Belirsiz/desteklenmeyen dil: {language}")
            return

        translate_started = time.perf_counter()
        translated = self.translator.translate(text)
        translate_ms = (time.perf_counter() - translate_started) * 1000
        total_ms = (time.perf_counter() - segment.started_at) * 1000
        print(f"[TR] {translated}")
        after_speech = ((time.perf_counter()-segment.ended_at)*1000
                        if segment.ended_at else 0)
        print(f"[SURE] ceviri={translate_ms:.0f} metin_hazir={total_ms:.0f}ms "
              f"konusma_sonundan={after_speech:.0f}ms | "
              f"kuyruk stt={AUDIO_QUEUE.qsize()} ceviri={self.translation_jobs.qsize()} "
              f"tts={self.speech_jobs.qsize()} oynatma={PLAY_QUEUE.qsize()}")
        self.submit_speech((translated, segment.started_at))

    def stop(self):
        self.tts.stop()
        self.whisper.stop()
        self.translator.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quality", "legacy"], default="quality")
    args = parser.parse_args()
    apply_profile(args.profile)
    print(f"\nLOCAL LIVE TRANSLATOR {APP_VERSION} - YEREL / AMD VULKAN")
    print(f"Model profili: {args.profile} | STT: {CFG.whisper_model.name} | ceviri: {CFG.llm_model.name}")
    input_device = select_device("input")
    output_device = select_device("output")
    CFG.output_device = output_device

    pipeline = player = capture = None

    def shutdown(*_):
        if not RUNNING.is_set():
            return
        RUNNING.clear()
        print("\nKapatiliyor...")
        for component in (capture, pipeline, player):
            if component is not None:
                component.stop()

    signal.signal(signal.SIGINT, shutdown)
    try:
        pipeline = Pipeline(output_device)
        player = Player(output_device)
        capture = Capture(input_device)
        pipeline.start()
        capture.start()
        print("\nHAZIR: Ingilizce yerelde cevrilir, Turkce dogrudan aktarilir. Cikis: CTRL+C")
        while RUNNING.is_set():
            time.sleep(0.1)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
