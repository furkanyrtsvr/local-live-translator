# Geliştirme ve test notları — 16 Ekim 2026

Bu belge ses modeli değişikliğinin teknik ayrıntılarını, ölçümleri ve bilinen
sınırlamaları açıklar. Genel kurulum ve kullanım için [README](../README.md).
17 Ekim'de yapılan VAD ve akış değişiklikleri için [güncel araştırma ve ölçüm notları](reference-projects.md).

## Çalıştırma

Depo kökünde `run_v3_local.bat` çalıştırılır. Açık eski çevirmeni önce kendi
penceresinde Ctrl+C ile kapatın; aynı çıkışta iki program birlikte konuşmasın.
Bu güncelleme C: çalışma klasöründedir; F: kopyası değiştirilmedi.

Piper yerine Supertonic 3 kullanılır. Model bir kez yüklenir, bütün konuşmalarda
aynı yerel Python işlemi kullanılır. Çalışma sırasında bulut, hesap veya API gerekmez.
Başlangıçta modeller ısıtılır; HAZIR yazısından sonra ses yakalama başlar.
Piper dosyaları silinmedi ancak V3 artık Piper kullanmıyor.

## Ses ayarları

`app/voice_config.json` değiştikten sonra uygulamayı yeniden başlatın:

- `provider`: `dml` AMD DirectML; `cpu` alternatif.
- `voice`: M1–M5 veya F1–F5; dosyalar yerelde hazır.
- `steps`: 8 varsayılan. Azaltmak hızı artırabilir ama kaliteyi düşürebilir.
- `speed`: 1.12 varsayılan; 1.0 daha yavaş. Perde değiştirilmeden modelin süre
  tahmini üzerinden uygulanır.

`samples/clean_voice_sample.wav` doğrulama komutuyla uygulamanın gerçek ses
işleme yolundan üretilir. Önceki yerel örnek dosyalar korunmuştur.
`samples/supertonic_cpu_F1_1.wav` alternatif kadın sesidir (ham model çıkışı).
`supertonic_worker.log` yalnızca yerel ses işleminin hata kaydıdır.

## Ölçümler / sınırlar

RX 7600, i5-14400F; 8 adım, hız 1.12, M1:

| Deney | CPU | DirectML |
|---|---:|---:|
| 3.58 saniyelik kısa örnek | 0.82 sn | ilk ölçüm 1.60 sn |
| 12.06 saniyelik paragraf | 2.05 sn | 1.10 sn |

Yeni DirectML oturumunda tekrar edilen kısa cümleler 0.87–1.13 sn sürdü.
Whisper ve Qwen de yüklüyken üç gerçek çeviri örneğinde TTS 0.68–0.86 sn sürdü.
Bu rakamlar **yalnızca sentez süresi**, uçtan uca gecikme değil. Ses yakalama,
cümle sonunu bekleme, STT, çeviri, kuyruk ve cihaz gecikmeleri ayrıca eklenir.
DirectML kısa cümlede CPU'dan daima hızlı değil. Ölçümler küçük örneklem;
oyun/GPU yükü altında tekrar ölçülmeli. Model cümleyi topluca üretir, gerçek
örnek-bazlı akış üretimi yapmaz. Önceki cümle çalınırken sonraki iş hazırlanabilir.

16 Ekim sürümünde 9 regresyon testi geçti: dil güveni, sessizlik, cümle sonlandırma, bağlam tekrarı,
çıkışta sessizlik, örnek kaybetmeden blok birleştirme, DC temizleme, gürültüyü
yükseltmeme ve anti-alias yeniden örnekleme.

## Dip ses

Çıkış callback'i boşta kesin sıfır üretir; açık cihazı verisiz bırakma kaldırıldı.
Doğrusal yeniden örnekleme yerine anti-alias polyphase filtre kullanılır.
Sentezlenen seste 45 Hz yüksek geçiren filtre, 4 ms uç yumuşatma ve yalnızca
aşırı tepe varsa aşağı yönde kazanç uygulanır. Sessiz sesi tepeye normalleştirip
dip gürültüyü büyütmez. Türkçe geçiş sesine bu TTS filtresi uygulanmaz.

Bunlar geniş bant hışırtıyı otomatik gideren bir denoiser değildir. Kullanıcının
kulaklık çıkışı kaydedilmedi/dinlenmedi; bildirilen dip sesin kaynağı kesinleşmedi.
WAV temiz ama canlı çıkış gürültülüyse Windows dinleme/monitoring, yinelenen
çıkış, ses cihazı ve analog bağlantı ayrıca incelenmelidir.

## Çeviri

Qwen3-4B Q4_K_M ve Argos yedeği değişmedi. Son testte:

EN: We should not jump to conclusions before seeing the results.

Qwen: Sonuçları göreceğe kadar karar vermeden önceden sonuçya varmamalıyız.

Bu açık bir kalite problemidir; daha iyi ses bozuk çeviriyi düzeltemez.
Çeviriye özel TranslateGemma 4B bir sonraki karşılaştırma adayıdır, kurulmuş veya
bu donanımda daha iyi olduğu doğrulanmış değildir. İndirme için resmî modelde
lisans kabulü gerekir. STT hâlâ Whisper base; kaynak metin hataları da mümkündür.

## Kaynak / yeniden kurulum

- Kaynak: https://github.com/supertone-oss-archive/supertonic
- Kaynak commit: `1e9799e964ea4c0dad7cde993b65c3c813a7b373`
- Model: https://huggingface.co/supertone-oss-archive/supertonic-3
- Model indirme revision: `aafc6e32416a594460b32413efc49d7fe4ce6d46`
- Kaynak MIT; model BigScience Open RAIL-M. Lisanslar vendor/model klasörlerinde.
- Resmî proje arşivlenmiş; bakım desteği yok. İndirilen dosyalar çevrimdışı çalışır.
- TranslateGemma: https://huggingface.co/google/translategemma-4b-it

Uygulama `py -3.13`; TTS ayrı `.venv-supertonic` Python 3.11 ortamında çalışır.
Mevcut Piper/Argos ONNX ortamını DirectML ile değiştirmemek için ayrılmıştır.

```cmd
py -3.13 -m pip install -r app/requirements-v3-local.txt
py -3.11 -m venv .venv-supertonic
.venv-supertonic\Scripts\python.exe -m pip install -r app/requirements-supertonic.txt
py -3.13 -m unittest discover -s app -p "test_*.py" -v
py -3.13 app/verify_local_voice.py
```

Bu komutlar depo kökünden çalıştırılır ve model/kaynak dosyalarının zaten mevcut
olduğunu varsayar. `app/verify_local_voice.py` mikrofonu açmadan ve kulaklıkta çalmadan gerçek modellerle
sentez, çeviri ve geri-transkripsiyon testi yapar.
