"""
=============================================================
  PC SESİ CANLI ÇEVİRMEN - V2.2
=============================================================
  Mimari:
    1. CAPTURE   → VB-Cable CABLE Output → sistem sesi yakalama
    2. VAD        → Ses aktivitesi tespiti
    3. STT        → faster-whisper (yerel, CPU int8)
    4. TRANSLATE  → deep-translator + rate-limit koruması
    5. TTS        → edge-tts (tr-TR-AhmetNeural)
    6. PLAYBACK   → sounddevice ile AÇIKÇA belirtilen çıkış cihazına
                    (VB-Cable'dan bağımsız, gerçek hoparlöre/kulaklığa)
=============================================================
"""

import queue
import threading
import time
import uuid
import os
import sys
import signal
import logging
import io
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ─── BAĞIMLILIK KONTROL ──────────────────────────────────────────────────────
def check_dependencies():
    missing = []
    for mod, pkg in [
        ("faster_whisper", "faster-whisper"),
        ("sounddevice",    "sounddevice"),
        ("edge_tts",       "edge-tts"),
        ("deep_translator","deep-translator"),
        ("pydub",          "pydub"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"EKSIK KUTUPHANE: {', '.join(missing)}")
        print(f"Kur: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import sounddevice as sd
import av
import io
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import edge_tts
import asyncio

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("translator")

# ─── YAPILANDIRMA ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    # STT
    # ".en" modelleri Türkçeyi algılayamaz. Çok dilli base modeli sayesinde
    # Türkçe ses çevrilmeden doğrudan çıkışa verilebilir.
    whisper_model:    str   = "base"
    whisper_device:   str   = "cpu"
    whisper_compute:  str   = "int8"
    whisper_language: Optional[str] = None  # Otomatik dil algılama
    whisper_beam_size:int   = 1           # beam_size=1 greedy: en hızlı
    cpu_threads:      int   = 8           # i5-14400F performans çekirdekleri

    # Ses yakalama (GİRİŞ — VB-Cable Output)
    sample_rate:            int   = 16000
    chunk_duration:         float = 0.10
    min_speech_duration:    float = 0.25
    first_segment_duration: float = 1.60
    max_segment_duration:   float = 2.40
    silence_threshold:      float = 0.0022  # Hassas ve güvenli eşik
    silence_duration:       float = 0.28

    # TTS çıkış cihazı (ÇIKIŞ — gerçek hoparlör/kulaklık)
    output_device_index: Optional[int] = None

    # TTS
    tts_voice: str = "tr-TR-AhmetNeural"
    tts_rate:  str = "+35%"

    # Çeviri
    translate_source: str = "en"
    translate_target: str = "tr"

    # Genel
    temp_dir:            str   = "."
    verbose:             bool  = True
    max_queue_size:      int   = 2
    silence_flush_after: float = 1.6      # Bu kadar sessizlik → kuyruğu temizle
    language_probability: float = 0.55
    passthrough_turkish:  bool  = True
    install_argos_model:  bool  = False  # Açılışta habersizce ~100 MB indirme yapma


config = Config()
running = True

# ─── GLOBAL QUEUE'LAR ────────────────────────────────────────────────────────
audio_queue:    queue.Queue = queue.Queue(maxsize=6)
playback_queue: queue.Queue = queue.Queue(maxsize=6)

last_audio_time: float = 0.0   # Son ses aktivitesi zamanı (sessizlik tespiti için)

stats = {"captured": 0, "translated": 0, "delay_total": 0.0}

# ─── VERİ SINIFLARI ──────────────────────────────────────────────────────────
@dataclass
class AudioSeg:
    data: np.ndarray
    t0:   float = field(default_factory=time.time)
    sid:  str   = field(default_factory=lambda: uuid.uuid4().hex[:6])

@dataclass
class PlayItem:
    audio_np:   np.ndarray   # float32, mono
    samplerate: int
    text_tr:    str
    text_en:    str
    delay_ms:   float
    created_at: float = field(default_factory=time.time)
    kind:       str = "translation"
    stt_ms:     float = 0.0
    translate_ms: float = 0.0
    tts_ms:     float = 0.0


# ─── SES YAKALAMA (GİRİŞ) ────────────────────────────────────────────────────
class AudioCapture:
    def __init__(self, device_index: Optional[int]):
        self.device_index = device_index
        self._buf  = np.array([], dtype=np.float32)
        self._preroll = np.array([], dtype=np.float32)  # Konuşma başını koruyan tampon (~0.25 sn)
        self._preroll_max = int(config.sample_rate * 0.25)
        self._rec_start:     Optional[float] = None
        self._silence_start: Optional[float] = None
        self._active = False
        self._is_first_segment = True                   # İlk cümlenin 2 saniyede çıkması için bayrak
        self._stream = None

    @staticmethod
    def _rms(x): return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0

    def _emit(self, now: float, keep_active: bool = False):
        """Mevcut tamponu kuyruğa yollar; callback thread'ini asla bloklamaz."""
        dur = len(self._buf) / config.sample_rate
        if dur >= config.min_speech_duration:
            seg = AudioSeg(data=self._buf.copy(), t0=self._rec_start or now)
            try:
                audio_queue.put_nowait(seg)
                stats["captured"] += 1
                if config.verbose:
                    print(f"  [YAKALANDI] {dur:.1f}sn ses -> isleme gonderildi...")
            except queue.Full:
                # Canlı akış geriye düşmüşse en eski segment yerine en yeniyi koru.
                try:
                    audio_queue.get_nowait()
                    audio_queue.put_nowait(seg)
                    print("  [UYARI] Islem kuyrugu geride; en eski segment atlandi")
                except queue.Empty:
                    pass

        self._buf = np.array([], dtype=np.float32)
        self._silence_start = None
        self._active = keep_active
        self._rec_start = now if keep_active else None
        self._is_first_segment = False

    def _cb(self, indata, frames, tinfo, status):
        global last_audio_time
        mono = indata[:,0] if indata.ndim > 1 else indata.flatten()
        rms  = self._rms(mono)
        now  = time.time()

        speech = rms > config.silence_threshold

        if speech:
            last_audio_time = now
            self._silence_start = None
            if not self._active:
                self._rec_start = now
                self._active = True
                # Pre-roll tamponunu cümlenin başına ekle (ilk harf/hece asla kesilmez)
                self._buf = np.concatenate([self._preroll, mono])
                self._preroll = np.array([], dtype=np.float32)
            else:
                self._buf = np.concatenate([self._buf, mono])

            # Eski kod bu kontrolü yalnızca sessizlik dalında yapıyordu. Bu yüzden
            # kesintisiz konuşma 10+ saniye boyunca hiç gönderilmeyebiliyordu.
            dur = len(self._buf) / config.sample_rate
            target_max = config.first_segment_duration if self._is_first_segment else config.max_segment_duration
            if dur >= target_max:
                self._emit(now, keep_active=True)

        elif self._active:
            self._buf = np.concatenate([self._buf, mono])
            if self._silence_start is None:
                self._silence_start = now

            dur      = len(self._buf) / config.sample_rate
            sil_dur  = now - self._silence_start
            
            # Doğal cümle/ifade sonunu kısa sessizlikte hemen gönder.
            target_max = config.first_segment_duration if self._is_first_segment else config.max_segment_duration
            flush = sil_dur >= config.silence_duration or dur >= target_max

            if flush:
                self._emit(now, keep_active=False)
        else:
            self._preroll = np.concatenate([self._preroll, mono])[-self._preroll_max:]
            if now - last_audio_time > 1.8:
                self._is_first_segment = True

    def start(self):
        self._stream = sd.InputStream(
            samplerate=config.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device_index,
            blocksize=int(config.sample_rate * config.chunk_duration),
            callback=self._cb,
            latency="low",
        )
        self._stream.start()
        name = sd.query_devices(self.device_index)["name"] if self.device_index is not None else "default"
        print(f"  Giris cihazi: [{self.device_index}] {name}")
        print(f"  Ses yakalama: AKTIF")

    def stop(self):
        if self._stream:
            try: self._stream.stop(); self._stream.close()
            except: pass


# ─── STT ─────────────────────────────────────────────────────────────────────
class STT:
    def __init__(self):
        print(f"  Whisper yukleniyor ({config.whisper_model}/{config.whisper_compute}, {config.cpu_threads} threads)...")
        self.model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute,
            cpu_threads=config.cpu_threads,
        )
        print("  Whisper: HAZIR")

    def warmup(self):
        """Modeli RAM'e ve CPU'ya tam olarak oturtur, ilk çalıştırma takılmasını önler."""
        dummy = np.zeros(int(config.sample_rate * 0.5), dtype=np.float32)
        try:
            segs, _ = self.model.transcribe(dummy, beam_size=1)
            list(segs)
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray, prompt: Optional[str] = None):
        segs, info = self.model.transcribe(
            audio,
            language=config.whisper_language,
            beam_size=config.whisper_beam_size,
            # Otomatik dil algılarken İngilizce prompt vermek sonucu İngilizceye
            # zorlayabilir. Prompt yalnızca dil sabitlenmişse kullanılmalı.
            initial_prompt=prompt if config.whisper_language and prompt else None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250, speech_pad_ms=150),
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, info.language, float(info.language_probability or 0.0)


# ─── ÇEVİRİ (Offline - argostranslate, Google fallback) ──────────────────────
class Translator:
    """
    Önce argostranslate (yerel, rate-limit yok) dener.
    Başarısız olursa Google'a düşer (yavaş, limitli).
    """
    def __init__(self):
        self._argos_ready = False
        self._google = GoogleTranslator(
            source=config.translate_source,
            target=config.translate_target,
        )
        self._google_lock = threading.Lock()
        self._google_last = 0.0
        self._setup_argos()

    def _setup_argos(self):
        try:
            from argostranslate import package, translate
            # Kurulu dil çiftlerini kontrol et
            installed = translate.get_installed_languages()
            en_lang = next((l for l in installed if l.code == "en"), None)
            tr_lang = next((l for l in installed if l.code == "tr"), None)

            if en_lang and tr_lang:
                translation = en_lang.get_translation(tr_lang)
                if translation:
                    self._argos_translate = translation.translate
                    self._argos_ready = True
                    print("  Ceviri: argostranslate (OFFLINE) - HAZIR")
                    return

            if not config.install_argos_model:
                print("  Ceviri: Argos modeli kurulu degil -> Google Translate kullanilacak")
                return

            # İstenirse model kurulu değilken indir
            print("  Ceviri: argostranslate en→tr modeli indiriliyor (~100MB)...")
            package.update_package_index()
            available = package.get_available_packages()
            pkg = next(
                (p for p in available if p.from_code == "en" and p.to_code == "tr"),
                None,
            )
            if pkg:
                pkg.install()
                installed = translate.get_installed_languages()
                en_lang = next((l for l in installed if l.code == "en"), None)
                tr_lang = next((l for l in installed if l.code == "tr"), None)
                if en_lang and tr_lang:
                    translation = en_lang.get_translation(tr_lang)
                    if translation:
                        self._argos_translate = translation.translate
                        self._argos_ready = True
                        print("  Ceviri: argostranslate HAZIR")
                        return
        except Exception as e:
            log.warning(f"argostranslate kurulum hatasi: {e}")

        print("  Ceviri: argostranslate kullanilamadi → Google Translate (online) kullanilacak")

    def translate(self, text: str) -> Optional[str]:
        # 1. Offline argostranslate
        if self._argos_ready:
            try:
                result = self._argos_translate(text)
                if result and result.strip().lower() != text.strip().lower():
                    return result.strip()
            except Exception as e:
                log.warning(f"argostranslate hatasi: {e}")

        # 2. Fallback: Google Translate (rate-limit korumalı)
        with self._google_lock:
            elapsed = time.time() - self._google_last
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)
            for attempt in range(2):
                try:
                    result = self._google.translate(text)
                    self._google_last = time.time()
                    if result and result.strip().lower() != text.strip().lower():
                        return result.strip()
                    return None
                except Exception as e:
                    log.warning(f"Google Translate hata (deneme {attempt+1}): {e}")
                    time.sleep(2.0)
        return None

    def warmup(self):
        """Çeviri modelini RAM'e yükler, ilk çevirideki donmayı önler."""
        try:
            self.translate("Hello and welcome.")
        except Exception:
            pass


# ─── TTS ─────────────────────────────────────────────────────────────────────
class TTS:
    async def _stream_synth(self, text: str) -> Optional[bytearray]:
        try:
            comm = edge_tts.Communicate(text, voice=config.tts_voice, rate=config.tts_rate)
            mp3_bytes = bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3_bytes.extend(chunk["data"])
            return mp3_bytes
        except Exception as e:
            log.warning(f"TTS hatasi: {e}")
            return None

    def synth_to_numpy(self, text: str) -> Optional[tuple]:
        """Edge-TTS'ten sesi RAM'de stream eder, PyAV ile C seviyesinde ~30ms'de doğrudan numpy dizisine cevirir."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mp3_bytes = loop.run_until_complete(self._stream_synth(text))
        finally:
            loop.close()

        if not mp3_bytes:
            return None

        try:
            target_sr = 44100
            if config.output_device_index is not None:
                try:
                    dev_info = sd.query_devices(config.output_device_index)
                    target_sr = int(dev_info.get("default_samplerate", 44100))
                except Exception:
                    pass

            container = av.open(io.BytesIO(mp3_bytes))
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)

            chunks = []
            for frame in container.decode(stream):
                for r in resampler.resample(frame):
                    chunks.append(r.to_ndarray()[0])

            if not chunks:
                return None

            samples = np.concatenate(chunks)
            return samples, target_sr
        except Exception as e:
            log.warning(f"PyAV decode hatasi: {e}")
            return None


# ─── OYNATICI (ÇIKIŞ) ────────────────────────────────────────────────────────
class Player:
    """
    sounddevice ile AÇIKÇA belirtilen çıkış cihazına çalar.
    VB-Cable'dan tamamen bağımsız → doğrudan hoparlör/kulaklığa.
    """
    def __init__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="player")
        self._thread.start()

    def _loop(self):
        while running or not playback_queue.empty():
            try:
                item: PlayItem = playback_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if config.verbose:
                    avg = stats["delay_total"] / max(stats["translated"], 1)
                    if item.kind == "passthrough":
                        print(f"\n  [TR DOGRUDAN | {item.delay_ms:.0f}ms]")
                    else:
                        print(f"\n  [toplam:{item.delay_ms:.0f}ms | ort:{avg:.0f}ms | "
                              f"STT:{item.stt_ms:.0f} ceviri:{item.translate_ms:.0f} TTS:{item.tts_ms:.0f}ms]")
                        print(f"  EN: {item.text_en}")
                        print(f"  TR: {item.text_tr}")

                sd.play(item.audio_np, samplerate=item.samplerate,
                        device=config.output_device_index, blocking=True)
            except Exception as e:
                print(f"  [HATA: Ses Oynatilamadi]: {e} (Cihaz No: {config.output_device_index})")

    def stop(self):
        try: sd.stop()
        except: pass


# ─── PİPELINE ────────────────────────────────────────────────────────────────
class Pipeline:
    """
    Tek thread pipeline: segmentleri geliş sırasıyla işler.

    Mantık:
      - audio_queue'dan segment al
      - STT → Translate → TTS → Playback queue
      - Türkçeyse çeviri/TTS yapmadan özgün sesi çıkışa ver
    """
    def __init__(self):
        self.stt = STT()
        self.tr  = Translator()
        self.tts = TTS()
        self._last_prompt = ""
        self._t  = threading.Thread(target=self._loop, daemon=True, name="pipeline")

    def start(self): self._t.start()

    def warmup(self):
        print("  Modeller RAM'e yuklenip isitiliyor (Warmup)...")
        self.stt.warmup()
        self.tr.warmup()
        print("  Modeller: ISITILDI (Ilk ceviri gecikmesi onlendi)")

    def _next_segment(self) -> Optional["AudioSeg"]:
        try:
            return audio_queue.get(timeout=0.4)
        except queue.Empty:
            return None

    def _loop(self):
        while running:
            seg = self._next_segment()
            if seg is None:
                continue
            self._proc(seg)

    @staticmethod
    def _output_rate() -> int:
        try:
            return int(sd.query_devices(config.output_device_index).get("default_samplerate", 44100))
        except Exception:
            return 44100

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate or len(audio) < 2:
            return audio.astype(np.float32, copy=False)
        count = max(1, round(len(audio) * target_rate / source_rate))
        old_x = np.arange(len(audio), dtype=np.float64)
        new_x = np.linspace(0, len(audio) - 1, count, dtype=np.float64)
        return np.interp(new_x, old_x, audio).astype(np.float32)

    @staticmethod
    def _enqueue(item: PlayItem):
        try:
            playback_queue.put(item, timeout=0.5)
        except queue.Full:
            print("  [UYARI] Oynatma yetisemedi; bu segment atlandi")

    def _proc(self, seg: "AudioSeg"):
        # 1. STT + otomatik dil algılama
        stt_started = time.perf_counter()
        try:
            text, language, language_probability = self.stt.transcribe(seg.data, prompt=self._last_prompt)
        except Exception as e:
            log.warning(f"STT hatasi: {e}"); return
        stt_ms = (time.perf_counter() - stt_started) * 1000

        if not text or len(text.strip()) < 2:
            return

        if config.verbose:
            print(f"  [DIL] {language} %{language_probability * 100:.0f}: {text}")

        # Türkçe kaynak ses: çeviri ve sentetik ses yok; özgün sesi doğrudan çal.
        if (config.passthrough_turkish and language == "tr" and
                language_probability >= config.language_probability):
            out_rate = self._output_rate()
            direct_audio = self._resample(seg.data, config.sample_rate, out_rate)
            self._enqueue(PlayItem(
                audio_np=direct_audio,
                samplerate=out_rate,
                text_tr=text,
                text_en="",
                delay_ms=(time.time() - seg.t0) * 1000,
                kind="passthrough",
                stt_ms=stt_ms,
            ))
            self._last_prompt = ""
            return

        # Yalnızca güvenilir biçimde İngilizce algılanan sesi çevir.
        if language != "en" or language_probability < config.language_probability:
            if config.verbose:
                print(f"  [ATLANDI] Dil Ingilizce/Turkce olarak guvenle belirlenemedi ({language})")
            self._last_prompt = ""
            return

        text_en = text
        self._last_prompt = text_en[-200:]

        # 2. Çeviri
        translate_started = time.perf_counter()
        text_tr = self.tr.translate(text_en)
        translate_ms = (time.perf_counter() - translate_started) * 1000
        if not text_tr:
            return

        # 3. TTS → numpy
        tts_started = time.perf_counter()
        result = self.tts.synth_to_numpy(text_tr)
        tts_ms = (time.perf_counter() - tts_started) * 1000
        if not result:
            return
        audio_np, sr = result

        # 4. Gecikme hesapla ve playback queue'ya ekle
        delay_ms = (time.time() - seg.t0) * 1000
        stats["translated"] += 1
        stats["delay_total"] += delay_ms

        item = PlayItem(
            audio_np=audio_np, samplerate=sr,
            text_tr=text_tr, text_en=text_en,
            delay_ms=delay_ms, created_at=time.time(),
            stt_ms=stt_ms, translate_ms=translate_ms, tts_ms=tts_ms,
        )
        self._enqueue(item)


# ─── YARDIMCILAR ─────────────────────────────────────────────────────────────
def _rm(path):
    try:
        if path and os.path.exists(path): os.remove(path)
    except: pass

def _cleanup_temp():
    for f in os.listdir(config.temp_dir):
        if f.startswith("tts_") and f.endswith(".mp3"):
            _rm(os.path.join(config.temp_dir, f))


# ─── CİHAZ SEÇİMİ ────────────────────────────────────────────────────────────
def pick_device(mode: str) -> Optional[int]:
    """
    mode: "input"  → VB-Cable Output (ses yakalama)
          "output" → Gerçek hoparlör/kulaklık (TTS çıkışı)
    """
    devs = sd.query_devices()
    print(f"\n{'='*64}")
    if mode == "input":
        print("  GIRIS cihazi secin (VB-Cable Output):")
    else:
        print("  CIKIS cihazi secin (hoparlor / kulaklık):")
    print("="*64)

    auto = None
    for i, d in enumerate(devs):
        ch_key = "max_input_channels" if mode == "input" else "max_output_channels"
        if d[ch_key] <= 0:
            continue
        name = d["name"]
        low  = name.lower()
        tag  = ""
        if mode == "input":
            if "cable output" in low and "vb-audio" in low and "point" not in low:
                tag = "  <-- [VB-CABLE]"
                if auto is None: auto = i
            elif "stereo" in low and ("mix" in low or "karim" in low):
                tag = "  <-- [STEREO MIX]"
                if auto is None: auto = i
        else:
            # Çıkış için: Realtek / dahili hoparlör otomatik seç
            if any(x in low for x in ["realtek", "speakers", "hoparlor", "headphone", "kulaklık"]) \
               and "cable" not in low and "droidcam" not in low:
                tag = "  <-- [HOPARLOR/KULAKLIK]"
                if auto is None: auto = i
        print(f"  [{i:2}] {name}{tag}")
    print("="*64)

    if auto is not None:
        print(f"\n  Otomatik: [{auto}] {devs[auto]['name']}")
        try:
            inp = input("  ENTER = kullan, baska numara gir: ").strip()
            return int(inp) if inp.isdigit() else auto
        except Exception:
            return auto
    else:
        try:
            inp = input("  Cihaz numarasi gir (ENTER = sistem varsayilani): ").strip()
            return int(inp) if inp.isdigit() else None
        except Exception:
            return None


def play_test_tone(dev_idx: Optional[int]) -> bool:
    try:
        dev_info = sd.query_devices(dev_idx)
        sr = int(dev_info.get("default_samplerate", 44100))
        ch = max(1, min(2, dev_info.get("max_output_channels", 2)))
        t = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 587.33 * t) + 0.1 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        if ch > 1:
            tone = np.column_stack([tone] * ch)
        sd.play(tone, samplerate=sr, device=dev_idx, blocking=True)
        return True
    except Exception as e:
        print(f"  Uyari: Bu cihaza test sesi gonderilemedi: {e}")
        return False


def pick_output_device() -> Optional[int]:
    while True:
        idx = pick_device("output")
        dev_name = sd.query_devices(idx)["name"] if idx is not None else "Varsayilan"
        print(f"\n  🔊 [{idx}] {dev_name} cihazina test sesi gonderiliyor...")
        play_test_tone(idx)
        ans = input("  Bip sesini kulaklik/hoparlorunuzden DUYDUNUZ MU? (ENTER=Evet, baska sayi=degistir): ").strip()
        if not ans or ans.lower() in ["e", "evet", "y", "yes"]:
            return idx
        if ans.isdigit():
            idx = int(ans)
            try:
                dev_name = sd.query_devices(idx)["name"]
                print(f"\n  🔊 [{idx}] {dev_name} cihazina test sesi gonderiliyor...")
                play_test_tone(idx)
                ans2 = input("  Bip sesini DUYDUNUZ MU? (ENTER=Evet, 'h'=listeye don): ").strip()
                if not ans2 or ans2.lower() in ["e", "evet", "y", "yes"]:
                    return idx
            except Exception as e:
                print(f"  Gecersiz cihaz: {e}")


# ─── BANNER ──────────────────────────────────────────────────────────────────
BANNER = """
================================================================
   PC SESI CANLI CEVIRMENI  v2.2
   [ VB-Cable/Loopback -> Whisper -> Translate -> Edge-TTS ]
================================================================"""


# ─── ANA PROGRAM ─────────────────────────────────────────────────────────────
def main():
    global running
    print(BANNER)
    print(f"  Ses    : {config.tts_voice}")
    print(f"  Model  : whisper/{config.whisper_model} ({config.whisper_compute})")
    print(f"  Hiz    : {config.tts_rate}")
    print("="*64)

    # 1. GİRİŞ cihazı: VB-Cable Output (sistem sesini yakala)
    in_dev = pick_device("input")

    # 2. ÇIKIŞ cihazı: Gerçek hoparlör/kulaklık (TTS çal)
    print("\n  DIKKAT: Cikis cihazi, ceviri sesini DUYACAGINIZ kulaklik/hoparlordur.")
    print("  VB-Cable veya CABLE ile BASLAYAN bir sey SECTIYSENIZ YANLIS!")
    out_dev = pick_output_device()
    config.output_device_index = out_dev

    print("\n  Bilesenleri baslatiliyor...")
    pipeline = Pipeline()
    capture  = AudioCapture(device_index=in_dev)
    player   = Player()

    pipeline.warmup()
    pipeline.start()

    try:
        capture.start()
    except Exception as e:
        print(f"\n  HATA: Giris cihazi acilamadi: {e}")
        sys.exit(1)

    out_name = sd.query_devices(out_dev)["name"] if out_dev is not None else "sistem varsayilani"
    print(f"  Cikis cihazi: [{out_dev}] {out_name}")
    print("\n" + "="*64)
    print("  SISTEM HAZIR!")
    print("  Ingilizceyi Turkceye cevirir; Turkce sesi dogrudan aktarir.")
    print("  Her cikista STT / ceviri / TTS gecikmeleri ayri gosterilir.")
    print("  Cikis: CTRL+C")
    print("="*64 + "\n")

    def _shutdown(sig, frame):
        global running
        running = False
        print("\n  Kapatiliyor...")
        capture.stop()
        player.stop()
        _cleanup_temp()
        cap = stats["captured"]
        tra = stats["translated"]
        avg = stats["delay_total"] / max(tra, 1)
        print(f"\n  Istatistikler: yakalanan={cap} cevrilen={tra} ort_gecikme={avg:.0f}ms")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    while running:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
