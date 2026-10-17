# Benzer projelerden çıkarılan uygulama kararları

İnceleme: 17 Ekim 2026. Amaç Windows + RX 7600 üzerinde yerel İngilizce→Türkçe
sesli çevirinin akışını iyileştirmek. Aşağıdaki projelerin duyurduğu hızlar bizim
donanımımızda doğrulanmış sonuçlar olarak kullanılmadı.

| Proje | İncelenen yaklaşım | Bu projede alınan karar |
|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) | Durum taşıyan ONNX konuşma olasılığı, ayrı başlangıç/bitiş eşikleri | RMS eşiği yerine CPU'da Silero; 512 örnek çerçeve, 64 örnek bağlam; Torch gerekmez |
| [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) | Konuşma algılama, ayrı ses/model işlemleri, gecikme sınırlarının yönetimi | Ses callback'inde yalnızca kopyalama/kuyruk; VAD ayrı iş parçacığında; giriş kesintisinde durum sıfırlama |
| [Whisper-Streaming](https://github.com/ufal/whisper_streaming) | LocalAgreement ile ardışık tanımalarda kararlı metni onaylama | Uygulanmadı: mevcut whisper.cpp sunucusunda kararlı sözcük zamanları ve tekrarları ayıklayan katman gerekiyor |
| [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | Modüler akış, ASR backend seçimi ve çeviri desteği | Tanıma ve çeviri ayrı, sıralı kuyruklara ayrıldı; uygulamayı bütünüyle bu kütüphaneye taşımadık |
| [Polyglot](https://github.com/etiennechabert/polyglot) | Sistem sesi, paralel çeviri ve görülebilir buffer/işlem durumu | Terminale aşama kuyrukları ve konuşma sonundan metin hazır olana kadarki süre eklendi |

Polyglot'un belgelenen GPU yolu CUDA ve RTX 5080 odaklıdır; RX 7600 için
doğrudan hızlandırılmış bir çözüm değildir. RealtimeSTT'nin önerilen genel GPU
yolu da faster-whisper/CUDA'dır. Bu yüzden yararlı akış ilkelerini kendi
Vulkan + DirectML yolumuza uyguladık. Dış projelerin uygulama kodu kopyalanmadı;
Silero modelinin ONNX giriş/çıkış sözleşmesine uygun küçük bir NumPy adaptörü yazıldı.

## Uygulanan değişiklikler

- Silero başlangıç eşiği 0.50, devam eşiği 0.35. Zayıf heceler korunur.
- Konuşma öncesi 192 ms tampon ve bitişte 96 ms ses kuyruğu korunur.
- Uzun parçada 256 ms, kısa parçada 384 ms duraklama beklenir.
- 6.5 saniyeyi geçen parçada 96 ms'lik duraklama kesmeye izin verir.
  Hiç duraklama yoksa 8 saniyede zorunlu bölünür. Anlamsal cümle sonu garantisi yoktur.
- En az 256 ms konuşma olasılığı gerektirilir; kısa darbeler STT'ye gönderilmez.
- Input callback'i model çalıştırmaz, büyüyen dizileri birleştirmez veya terminale yazmaz.
- STT, çeviri ve TTS ayrı iş parçacıklarında; çeviri bağlamı ve ses sırası korunur.
- Uzun TTS metni tamamlanmış cümle gruplarıyla üretilir; ilk grup oynatmaya
  gönderilirken sonraki hazırlanabilir. Kısa cümleler birlikte tutulur.
- Kuyruk taşmaları ve 12 saniyeyi aşan ses hazırlama gecikmesi görünürdür.
  Ağır yük altında konuşma hızına yetişme garantisi verilmez.

## Gerçek modelle ölçüm

RX 7600 / i5-14400F, Supertonic 3 DirectML, 8 adım, hız 1.12.
Test sırasında başka model yükleme işi de vardı; sonuçlar küçük örneklemdir.

| Deney | Sonuç |
|---|---|
| Silero, 32 ms çerçeve | Ortanca ~0.23 ms; yüzde 95 ~0.29–0.37 ms |
| 5 sn sessizlik / beyaz gürültü / 180 Hz ton | Üçünde de 0 konuşma parçası |
| Zira ile hazırlanmış İngilizce WAV | 1 konuşma parçası; VAD'nin son konuşma çerçevesinden çıkış kararına 256 ms |
| Uzun Türkçe metin: tüm metin hazır olana kadar bekleme | 1.40 sn ortanca; 3 deneme |
| Aynı metin: ilk cümle grubunu önce gönderme | 1.19 sn ortanca; 3 deneme |
| Gerçek VAD → Whisper → Qwen → Supertonic → oynatma kuyruğu | Başarılı; STT ~345 ms, çeviri ~414 ms, TTS ~1229 ms |

TTS karşılaştırması yaklaşık %15 daha erken ilk ses üretimi gösteriyor. Uçtan
uca kulaklık gecikmesi ölçülmedi. Testte canlı ses kaydı/oynatma açılmadı.
Bu beyaz gürültü/ton deneyi müzik, kalabalık, aksan veya oyun sesi değerlendirmesi değildir.

Yerel rapor: `samples/realtime-benchmark.json` (Git'e eklenmez).
Tekrar üretmek için 16-bit PCM İngilizce konuşma WAV'ı kullanın:

```cmd
py -3.13 app/benchmark_realtime.py --input "C:\path\english-speech.wav" --tts
py -3.13 app/verify_pipeline.py --input "C:\path\english-speech.wav"
py -3.13 -m unittest discover -s app -p "test_*.py" -v
```

## Çeviri kalitesi deneyi

Mevcut bağlamlı Qwen istemi, kısa ve deterministik bir İngilizce→Türkçe istemiyle
dört cümlede karşılaştırıldı. Yeni aday bazı cümlelerde İngilizce/Portekizce
sözcükler ekledi, bir olumsuz ifadeyi de bozdu. Bu nedenle aday üretime alınmadı.
Mevcut modelde de bozuk Türkçe ve anlam hataları devam ediyor. İstem kısaltmak
tek başına kalite çözümü olmadı.

Sonraki büyük çalışmalar: Türkçe çeviri modelini sabit bir değerlendirme setiyle
karşılaştırmak; uzun kesintisiz konuşmada kararlı kısmi transkripsiyon;
gerçek oyun/film sesiyle uçtan uca gecikme ve anlaşılabilirlik ölçmek.

## Sürümler ve kaynaklar

- Silero kaynak/model commit: `1e261b036686cd0017d500ee96acd1c4ba572a9d`.
- Model SHA256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
- [Silero ONNX adaptör referansı](https://github.com/snakers4/silero-vad/blob/1e261b036686cd0017d500ee96acd1c4ba572a9d/src/silero_vad/utils_vad.py).
- [Silero MIT lisansı](https://github.com/snakers4/silero-vad/blob/1e261b036686cd0017d500ee96acd1c4ba572a9d/LICENSE).
- RealtimeSTT inceleme commit'i: `8918be3b2c88601f9d7276259863c544a654d633`.
- Whisper-Streaming inceleme commit'i: `6da90b44b7e50d79695e68166d2a2c7609c75abb`.
- WhisperLiveKit inceleme commit'i: `363e4f6d029694d9c81ae548beddd9d3c88a3637`.
