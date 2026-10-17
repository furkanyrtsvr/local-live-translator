# Local Live Translator

Windows'ta bilgisayar sesini yakalayıp İngilizce konuşmaları yerel modellerle
Türkçe seslendiren deneysel uygulama. Türkçe olarak algılanan parçalar yeniden
seslendirilmeden çıkışa aktarılır. README Türkçedir; proje adı evrensel kullanım
için İngilizce tutulmuştur.

> Geliştirme aşamasında: sıfır gecikmeli simultane çeviri değildir. Çeviri
> doğruluğu, cümle sınırları ve kuyruk gecikmesi hâlâ iyileştirme gerektirir.

## Güncel sürüm: v0.4.0-rc.1

Bu sürüm aday sürümdür. Önceki sürümler [GitHub Releases](https://github.com/furkanyrtsvr/local-live-translator/releases)
altında korunur. [Sürüm notları ve gecikme ölçümleri](docs/releases/v0.4.0-rc.1.md).

- **Ses yakalama:** sounddevice ve VB-CABLE.
- **Konuşma algılama:** yerel Silero VAD (CPU/ONNX), ayrı ses işleme iş parçacığı.
- **Konuşma tanıma / dil algılama:** whisper.cpp, Whisper small, Vulkan.
- **Çeviri:** llama.cpp üzerinde TranslateGemma 4B Q5_K_M; Argos en→tr yedeği.
- **Seslendirme:** Supertonic 3; AMD DirectML veya CPU, ayrı Python işlemi.
- **Ses çıkışı:** sürekli açık callback akışı, boşta dijital sessizlik,
  anti-alias yeniden örnekleme ve sentezlenen seste uç yumuşatma.

V3, bağımlılıklar ve modeller indirildikten sonra yerelde çalışır. Model
sunucuları yalnızca `127.0.0.1` üzerinde dinler; bir bulut çeviri/TTS hesabı gerekmez.
`app/translator_v2.py` eski sürümdür: Google/Edge tabanlı ağ servisleri kullanır ve
V3'ün çevrimdışı çalışma özelliğine sahip değildir.

## Dosya düzeni

```text
local-live-translator/
├── app/                     # Uygulama, testler, bağımlılıklar ve ses ayarı
├── docs/
│   └── development-notes.md  # Teknik değişiklikler ve test sonuçları
├── README.md
├── run_v3_local.bat          # Başlatıcı
└── .gitignore
```

Yerel `models/`, `vendor/`, `samples/` ve sanal ortamlar depo kökünde kalır;
GitHub'a yüklenmez. Aşağıdaki komutlar depo kökünden çalıştırılır.

## Gereksinimler

Test edilen ortam: Windows, Intel i5-14400F, AMD Radeon RX 7600, yaklaşık 16 GB RAM.
Diğer donanımlarda hız garantisi yoktur.

- Python 3.13: ana uygulama (`py -3.13`).
- Python 3.11: izole Supertonic ortamı.
- Git; Vulkan destekli ekran kartı sürücüsü.
- PC sesini yönlendirmek için [VB-CABLE](https://vb-audio.com/Cable/).
- Aşağıdaki model/binary dosyaları için birkaç GB boş disk alanı.

## Kurulum

Depoyu klonlayıp klasöründe CMD açın:

```cmd
git clone https://github.com/furkanyrtsvr/local-live-translator.git
cd local-live-translator
py -3.13 -m pip install -r app/requirements-v3-local.txt
py -3.11 -m venv .venv-supertonic
.venv-supertonic\Scripts\python.exe -m pip install -r app/requirements-supertonic.txt
```

Python paketlerini kurmak tek başına yeterli değildir: büyük modeller ve
üçüncü taraf binary'ler bu depoya yüklenmemiştir.

### Modeller ve yerel sunucular

Dosyaları aşağıdaki yerleşime hazırlayın. ZIP içindeki DLL'leri de ilgili
sunucunun yanında tutun; yalnızca EXE'yi kopyalamayın.

| Bileşen | Beklenen yol | Kaynak |
|---|---|---|
| Silero VAD | `models/vad/silero_vad.onnx` | `py -3.13 app/download_vad.py` ile sabit sürüm ve SHA256 doğrulaması |
| Whisper Vulkan sunucusu | `vendor/whisper-vulkan/whisper-server.exe` | [whisper.cpp-rocm v1.8.4](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/tag/v1.8.4) Windows Vulkan paketi |
| Whisper small | `models/whisper/ggml-small.bin` | [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp); indirme komutu aşağıda |
| llama.cpp Vulkan sunucusu | `vendor/llama-vulkan/llama-server.exe` | [llama.cpp b11696](https://github.com/ggml-org/llama.cpp/releases/tag/b11696) Windows Vulkan paketi |
| TranslateGemma 4B Q5 | `models/llm/translategemma-4b-it.Q5_K_M.gguf` | [GGUF ağırlıkları](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF); indirme komutu aşağıda |
| Supertonic kaynak | `vendor/supertonic/py/helper.py` | Aşağıdaki sabit commit |
| Supertonic ONNX ve sesler | `models/supertonic-3/onnx/`, `models/supertonic-3/voice_styles/` | Aşağıdaki model indirmesi |

Supertonic kaynak ve ağırlıkları:

```cmd
git clone https://github.com/supertone-oss-archive/supertonic.git vendor/supertonic
git -C vendor/supertonic checkout 1e9799e964ea4c0dad7cde993b65c3c813a7b373
py -3.13 -m pip install huggingface_hub
py -3.13 -c "from huggingface_hub import snapshot_download; snapshot_download('supertone-oss-archive/supertonic-3', revision='aafc6e32416a594460b32413efc49d7fe4ce6d46', local_dir='models/supertonic-3', allow_patterns=['onnx/*','voice_styles/*','LICENSE','README.md'])"
```

Argos İngilizce→Türkçe paketi de kurulu olmalıdır. Bu adım internet kullanır:

```cmd
py -3.13 -c "import argostranslate.package as p; p.update_package_index(); package=next(x for x in p.get_available_packages() if x.from_code=='en' and x.to_code=='tr'); p.install_from_path(package.download())"
py -3.13 app/download_vad.py
py -3.13 app/download_quality_models.py
```

Bu yönergeler mevcut çalışma ortamındaki düzeni belgeler; sıfırdan temiz bir
Windows kurulumunda uçtan uca kurulum testi henüz yapılmamıştır.

## Kullanım

1. Çevrilecek uygulamanın Windows ses çıkışını **CABLE Input** olarak seçin.
2. `run_v3_local.bat` dosyasını çalıştırın.
3. Çevirmenin girişinde **CABLE Output**, çıkışında gerçek kulaklık/hoparlör seçin.
4. **HAZIR** mesajını bekleyin. Kapatmak için Ctrl+C kullanın.

Çevirmenin çıkışını tekrar CABLE'a göndermeyin; geri besleme oluşabilir.
Aynı anda eski ve yeni sürümü çalıştırmayın. Türkçe doğrudan geçiş de dil
algılamasını beklediğinden gecikmesiz değildir.

Varsayılan `quality` profili Whisper small + TranslateGemma kullanır. İndirme
komutu bu iki dosyayı sabit revision ve yayınlanan SHA256 ile doğrular.
Eski modeller diskte varsa aşağıdaki komutla eski model profili seçilebilir:

```cmd
run_v3_local.bat --profile legacy
```

`legacy` profili Whisper base + Qwen3-4B dosyalarını kullanır; silmez veya yeniden
indirmez. Uygulamanın tamamının eski kaynak kodu için ilgili Release'i indirin.

## Ses ayarları

`app/voice_config.json` değiştikten sonra yeniden başlatın:

```json
{
  "provider": "dml",
  "voice": "M1",
  "steps": 8,
  "speed": 1.12
}
```

`provider`: AMD için `dml`, alternatif olarak `cpu`. `voice`: M1–M5 veya F1–F5.
`speed`: 1.0 normal, 1.12 daha hızlı. Daha az `steps` kaliteyi düşürebilir.
Piper dosyaları V3 için artık gerekli değildir.

Canlı akışta sesin süre tahmini kaynak parçanın süresiyle karşılaştırılır.
Gerekirse `speed` değerinden başlayarak `max_speed` sınırına kadar hızlandırılır
(varsayılan `app/voice_config.json`: 1.55). Ton perdesi yeniden örneklemeyle
değiştirilmez; modelin süre kontrolü kullanılır. Kısa konuşmalar normal hızda kalabilir.
Kullanıcının seçtiği politika cümleleri korumaktır; geride kaldığında özetleme
veya eski cümleleri atma yapılmaz. Ağır yükte gecikme yine büyüyebilir.

## Doğrulama ve örnek ses

Bağımlılıklar ve modeller hazırken:

```cmd
py -3.13 -m unittest discover -s app -p "test_*.py" -v
py -3.13 app/verify_local_voice.py
.venv-supertonic\Scripts\python.exe app/benchmark_supertonic.py --provider dml --voices M1 F1
```

`app/verify_local_voice.py` mikrofon kaydı veya kulaklıkta oynatma yapmadan gerçek
modelleri sınar ve `samples/clean_voice_sample.wav` üretir. Örnek sesler, loglar,
modeller ve sanal ortamlar Git tarafından dışlanır.

Test bilgisayarında üç örnekte TTS 0,68–0,86 saniye sürdü. Bu **yalnızca ses
üretimi**, uçtan uca gecikme değildir. Kısa cümlelerde CPU bazen DirectML'den
hızlı olabilir. Ayrıntılar: [ölçüm ve geliştirme notları](docs/development-notes.md).

17 Ekim akış geliştirmeleri, benzer projeler ve karşılaştırmalı ölçümler:
[reference-projects.md](docs/reference-projects.md). Silero VAD, ayrı STT/çeviri
kuyrukları ve uzun metinde ilk cümle grubunun daha erken seslendirilmesi eklendi.

## Bilinen sınırlamalar

- Konuşma parçaları Silero olasılıkları ve duraklamayla ayrılır; anlamsal cümle
  sonu garantisi yoktur. İlk kesintisiz parça 3.2 saniyede, sonraki parçalar
  4 saniyede bölünür. Sınırda 640 ms ses bağlamı korunur; tekrar sözcükleri ayıklanır.
  Yoğun akışta kuyruk büyümesi hâlâ mümkündür. Bekleyen STT parçaları bellekte
  korunur; uzun süre yetişemeyen kullanımda bellek tüketimi artar.
- TranslateGemma küçük testte bazı belirgin Qwen hatalarını azalttı; deyim,
  ayrıntı ve vurgu kayıpları devam edebilir. Genel doğruluk puanı hesaplanmadı.
- `quality` daha yavaştır: daha büyük modeller düşük gecikme garantisi değildir.
- Kendi TTS sesimiz girişte tekrar tanınırsa yüksek akustik benzerliğe sahip
  Türkçe parçalar tekrar oynatılmaz ve `[SES DONUSU]` uyarısı gösterilir. Bu bir
  tam yankı giderici değildir; yanlış Windows ses yönlendirmesi düzeltilmelidir.
- Supertonic cümle grubunu topluca üretir; uzun metinde gruplar sırayla gönderilir.
  Modelin içinde örnek-bazlı gerçek zamanlı üretim yapılmaz.
- Dip ses kaynağı donanım/yönlendirme olabilir. Filtre geniş bant gürültü giderici değildir.
- Supertonic'in resmî deposu arşivlenmiştir; güncelleme desteği yoktur.

## Üçüncü taraf lisansları

Supertonic kaynak kodu MIT, model ağırlıkları BigScience Open RAIL-M kapsamındadır.
Silero VAD MIT lisanslıdır; kaynak/model sürümü [araştırma notlarında](docs/reference-projects.md) belirtilmiştir.
Whisper, llama.cpp, Qwen, Argos ve diğer bağımlılıkların kendi lisansları geçerlidir.
TranslateGemma ağırlıkları Gemma kullanım koşullarına tabidir.
Bu depo üçüncü taraf model/binary dosyalarını yeniden dağıtmaz. Proje kodu için
ayrıca bir açık kaynak lisansı henüz seçilmemiştir.
