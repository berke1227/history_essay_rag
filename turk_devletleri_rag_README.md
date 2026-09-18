# Türk Devletleri RAG

Bir klasördeki (~13) Türk devletleri makalesini (PDF) okuyup vektör
veritabanında indeksleyen; sorulan sorulara **yalnızca bu makalelere
dayanarak** cevap veren; cevabı üretip kendi kendine doğrulayan ve
kaynak dışına çıkmadan "bu konu makalelerde yok" diyebilen bir
soru-cevap sistemi.

## Mimari

```
makaleler/*.pdf ──▶ PDF çıkarma ──▶ chunking (kelime penceresi + overlap)
                     (pdfplumber)         │
                                          ▼
                              ChromaDB (turk_devletleri koleksiyonu)
                                          │
soru ──────────────────────────────────▶ retrieval (eşik + dedup filtreli)
                                          │
                              ┌───────────▼────────────┐
                              │   ÜRET (LLM)            │◀──┐
                              │   sadece verilen        │   │ geri bildirimle
                              │   alıntılara dayanarak   │   │ yeniden dene
                              └───────────┬────────────┘   │
                                          ▼                │
                              ┌───────────────────────┐    │
                              │   DOĞRULA (LLM)        │────┘ (max N deneme)
                              │   grounded? yeterli    │
                              │   uzunlukta mı?         │
                              └───────────┬────────────┘
                                          ▼ (geçti veya deneme tükendi)
                                       Cevap
```

LLM adımları (üret + doğrula) yerel **`ollama`** (`qwen3.5:4b`) veya
ağa çıkmayan `mock` sağlayıcısı ile çalışır (bkz. `src/tdrag/providers/`).
Model seviyesinde thinking/reasoning kapatılmış ve yanıtlar filtrelenmiştir.

## Kurulum

```bash
uv sync --extra dev          # temel + test bağımlılıkları
# Sentence-transformer embedding desteği için:
uv sync --extra dev --extra embeddings
cp .env.example .env         # ve .env dosyasını doldurun
```

## Kullanım

1. Makalelerinizi `makaleler/` klasörüne PDF olarak koyun.
2. Ollama'da modelin kurulu olduğundan emin olun:
   ```bash
   ollama run qwen3.5:4b
   ```
3. `.env` dosyasında ayarları kontrol edin (varsayılan: `ollama` + `qwen3.5:4b`).
3. Çalıştırın:

```bash
uv run tdrag
```

İlk çalıştırma `makaleler/` klasörünü indeksler (tekrar çalıştırmak
mevcut dosyaları yeniden indekslemez — bkz. "Idempotent ingestion").
Ardından soru sorabileceğiniz etkileşimli bir döngü açılır; çıkmak
için `q`.

### Test verisi

`makaleler/` klasöründeki 3 PDF **gerçek akademik makale değildir** —
yalnızca boru hattını (PDF çıkarma → chunking → embedding → retrieval)
uçtan uca deneyebilmeniz için üretilmiş, temel düzeyde ve tartışmasız
üç kısa metindir (Göktürk Kağanlığı, Osmanlı'nın kuruluşu, Büyük
Selçuklu Devleti). Kendi makalelerinizi eklediğinizde bunları silebilir
veya üzerine yazabilirsiniz. Yeniden üretmek isterseniz:

```bash
uv run python scripts/generate_sample_pdfs.py
```

## Testler

```bash
uv run pytest -v
```

46 test; `chunker`, `pdf_loader`, `config` doğrulamaları, `vector_store`
(jaccard/dedup/eşik/upsert), `qa_pipeline` (üret→doğrula döngüsünün
tüm dalları: ilk denemede geçer, ikinci denemede geçer, hep reddedilir,
doğrulayıcı bozuk JSON döner, kod-seviyesi uzunluk kontrolü) ve
`ingestion` (kısmi hata toleransı) kapsanır.

Testler **gerçek bir LLM'e veya internete bağlanmaz**: `MockLLMProvider`
(scripted mod) ve `HashingEmbeddingFunction` (model indirmeyen,
deterministik embedding) kullanılır. Bu, üretimdeki gerçek sağlayıcı
kodunun (ör. Anthropic SDK çağrısının) *kendisini* değil, boru
hattındaki orkestrasyon mantığını (yeniden deneme, eşikleme, hata
yönetimi) doğrular — sağlayıcı sınıfları (`providers/*.py`) küçük ve
SDK'ya ince bir sarmalayıcı olacak şekilde kasıtlı olarak basit
tutulmuştur.

## Mimari kararlar (belgelendirilmiş sapmalar)

Talimattaki gereksinimlerin bazıları, sonsuz döngü / öngörülemez
davranış gibi riskler taşıdığı için sınırlandırılmış ve burada
belgelenmiştir:

- **"En gerçekçi cevap alana kadar kendini yenileyecek" → sınırlı
  döngü.** `TDRAG_MAX_VERIFICATION_ATTEMPTS` (varsayılan 3) ile
  sınırlıdır. Sınıra ulaşılırsa son cevap `grounded=False` bayrağıyla
  döner; sessizce başarılı gösterilmez. Sebep: doğrulayıcı asla tatmin
  olmazsa sınırsız döngü sonsuz maliyet/gecikme riski taşır.
- **Chunking: paragraf-farkında değil, kelime penceresi (sliding
  window).** Kelime sınırında keser (asla kelimeyi bölmez), çok uzun
  tek paragraflarda özel durum kodu gerektirmez, davranışı test etmesi
  kolaydır. `chunker.py` docstring'inde detaylandırılmıştır.
- **OCR desteklenmiyor.** PDF'te metin katmanı yoksa (taranmış görüntü)
  `pdf_loader` bunu sessizce boş geçmek yerine açık bir hata ile
  bildirir. Akademik makalelerin çoğu doğrudan metin katmanlı
  (born-digital) PDF'ler olduğundan bu makul bir kapsam sınırıdır;
  gerekirse `pytesseract` ile ayrı bir adım olarak eklenebilir.
- **similarity_threshold embedding modeline bağlıdır — kalibrasyon
  gerekir.** Varsayılan (0.25), üretim için önerilen
  `SentenceTransformerEmbeddingFunction` için kabaca uygun bir
  başlangıç noktasıdır. Testlerdeki `HashingEmbeddingFunction` çok
  daha düşük mutlak skorlar ürettiğinden testler ayrı (düşük) bir eşik
  kullanır (bkz. `tests/conftest.py`). Kendi 13 makalenizle birkaç soru
  deneyip gerekirse `TDRAG_SIMILARITY_THRESHOLD`'u ayarlayın: çok
  düşükse alakasız sorulara da cevap üretilir, çok yüksekse alakalı
  sorular da "kapsam dışı" sayılır.
- **Idempotent ingestion: hash kontrolü yerine `upsert`.** Aynı dosya
  tekrar işlenirse (deterministik `dosya_adi::chunk_index` id'si
  sayesinde) kayıt çoğalmaz, üzerine yazılır. Ayrı bir "zaten var mı"
  kontrol mekanizmasına gerek bırakmaz.
- **Retrieval'da mükerrer parça filtreleme.** Örtüşen (overlap)
  pencereler nedeniyle neredeyse aynı içerikli iki parça aynı sorguya
  yüksek skorla dönebilir; Jaccard kelime-kümesi benzerliği
  `TDRAG_DEDUP_JACCARD_THRESHOLD` üzerindeyse ikincisi elenir.
- **Kod seviyesinde "en az 1 paragraf" kontrolü (defense in depth).**
  Doğrulayıcı LLM "gecti: true" dese bile, `qa_pipeline` ayrıca kelime
  sayısını kontrol eder — LLM'in uzunluk kriterini gözden kaçırma
  ihtimaline karşı ikinci bir güvenlik katmanı.

## Proje yapısı

```
src/tdrag/
├── config.py          # doğrulamalı, ortam değişkeninden okunabilir ayarlar
├── models.py           # Chunk / RetrievedChunk / VerificationResult / Answer
├── pdf_loader.py        # PDF -> metin (pdfplumber, OCR yok)
├── chunker.py           # metin -> kelime pencereli chunk'lar
├── embeddings.py         # HashingEmbeddingFunction (test) / SentenceTransformer (üretim)
├── vector_store.py       # ChromaDB sarmalayıcı: upsert, eşik+dedup filtreli query
├── prompts.py            # cevap ve doğrulama prompt şablonları
├── qa_pipeline.py         # getir -> üret -> doğrula döngüsü
├── ingestion.py            # klasör -> chunk -> vector store akışı
├── cli.py                   # etkileşimli giriş noktası
└── providers/
    ├── base.py               # LLMProvider Protocol
    ├── mock_provider.py        # ağa çıkmayan test/demo sağlayıcı
    └── ollama_provider.py       # yerel Ollama (thinking/reasoning kapalı)

tests/            # 46 test, tamamı mock/hashing ile (ağa çıkmaz)
scripts/generate_sample_pdfs.py    # sentetik test PDF üretici
makaleler/                         # PDF'lerinizin gideceği yer
```

## Bilinen sınırlamalar

- OCR yok (yukarıda açıklandı).
- `similarity_threshold` kendi korpusunuzla elle kalibre edilmelidir;
  otomatik kalibrasyon yapılmaz.
- Sağlayıcı SDK'larının gerçek API çağrıları bu ortamda ağ kısıtlaması
  nedeniyle canlı test edilememiştir (bkz. "Testler"); `ollama`,
  `openai`, `gemini` sağlayıcıları kendi ortamınızda ilk kullanımda
  ayrıca doğrulanmalıdır.
