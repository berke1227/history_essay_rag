# HISTORIA AQUILAE (TDRAG)
### *De Re Historica • Fridericus II • Kartalın Tarihi*

Kutsal Roma İmparatoru ve Sicilya Kralı **II. Friedrich**'in ilim ve şahin avı mirası ekseninde tasarlanmış; Türk devletleri ve Ortaçağ Akdeniz/Sicilya-Norman tarihi makaleleri (PDF) üzerinde çalışan, **sıfır halüsinasyon hedefli**, kendi kendini doğrulayan (self-correcting) ve kaynak sadakatine tam bağlı akademik RAG (Retrieval-Augmented Generation) soru-cevap platformu.

---

## 🦅 Öne Çıkan Özellikler

- **Sıkı Kaynak Sadakati (Zero-Hallucination):** Cevaplar yalnızca yüklenen PDF makalelerindeki vesikalara dayanır. Kaynaklarda yer almayan hiçbir dış bilgi cevaba karıştırılmaz; konu korpusta yoksa açıkça *"Bu konu makalelerde yer almamaktadır"* uyarısı verilir.
- **Konu Sapmasını (Drift) Önleyen Guardrails:** Farklı devlet veya dönemlere ait sorgularda (örneğin Normanlar makalesi sorgulanırken Harezmşahlar veya Selçuklulara kayma gibi) retrieval filtreleri ve odak kontrolleri ile konu bütünlüğü korunur.
- **6K Geniş Bağlam Penceresi (6144 Token):** `qwen3.5:4b` modeli için 6K bağlam penceresi ile çok sayfalı zengin tarihsel alıntılar tek seferde işlenir.
- **500 Token Çıktı Sınırı:** Cevapların yarıda kesilmesini önleyen, geniş ve detaylı analitik açıklamaları destekleyen 500 token çıkış kapasitesi.
- **The House of Da Vinci 3 Estetiğinde Web Arayüzü:** Koyu obsidyen, antik altın varak ve cam pencereler (`backdrop-filter: blur(16px)`), *Cinzel* ve *Inter* tipografisi.
- **İnteraktif Kartal İniş Sahnesi:** Sayfa açılışında Castel del Monte kalesi önünde gökyüzünde süzülen kartal; *"Kartalı Koluna İndir & Keşfe Başla"* dendiğinde pürüzsüzce II. Friedrich'in deri eldivenli koluna konar, arayüz çalışma istasyonuna dönüşür ve sahne kalıcı canlı arka plan olarak kalır. Üst çubuktaki buton ile dilediğiniz an sahne yeniden oynatılabilir.
- **Katlanabilir "Makaleler" Kütüphanesi:** Sol menüdeki makale başlığına tıklandığında liste ve arama alanı zarif bir akordeon animasyonuyla gizlenip açılır; kullanıcının tercihi saklanır.
- **Sürükle-Bırak PDF Yükleme & Makale Odaklama:** Web arayüzünden doğrudan yeni PDF vesikaları yüklenebilir veya sol menüden tek bir makaleye odaklanarak özel araştırma yapılabilir.
- **Çift Kullanım Modu:** İster modern Web Arayüzü (`http://localhost:8000`), ister etkileşimli Terminal CLI.

---

## 🏛️ Mimari Akış

```text
  ┌───────────────────────────────────────────────────────────┐
  │                    PDF Yükleme / İndeks                   │
  │   makaleler/*.pdf ──▶ pdfplumber ──▶ Sliding Window       │
  │                                      (Kelime penceresi)   │
  └─────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │          ChromaDB Vektör Veritabanı (.chroma/)            │
  │    (paraphrase-multilingual-MiniLM-L12-v2 Embedding)      │
  └─────────────────────────────┬─────────────────────────────┘
                                │
   Kullanıcı Sorusu ────────────▶ Retrieval (Eşik + Jaccard Dedup Filtresi)
                                │
                                ▼
                     ┌─────────────────────┐
                     │   1. ÜRET (LLM)     │◀──┐
                     │   Sadece verilen    │   │
                     │   parçalara dayalı  │   │ Geri bildirimle
                     └──────────┬──────────┘   │ yeniden üret
                                ▼              │ (Max 3 Deneme)
                     ┌─────────────────────┐   │
                     │   2. DOĞRULA (LLM)  │───┘
                     │   Grounded? Uydurma │
                     │   bilgi var mı?     │
                     └──────────┬──────────┘
                                ▼ (Doğrulamadan Geçti)
                 Nihai Cevap + Sayfa Kaynakçası
```

---

## 🛠️ Kurulum

Gereksinimler: **Python 3.11+**, [uv](https://docs.astral.sh/uv/) ve yerel çalışan [Ollama](https://ollama.com/).

```bash
# Depoyu klonlayın
git clone https://github.com/berke1227/history_essay_rag.git
cd history_essay_rag/turk_devletleri_rag

# Bağımlılıkları yükleyin (geliştirme ve embedding paketleri dahil)
uv sync --extra dev --extra embeddings

# Örnek çevre değişkenlerini kopyalayın
cp .env.example .env
```

Ollama üzerinde modelin kurulu olduğundan emin olun:
```bash
ollama run qwen3.5:4b
```

---

## 🚀 Çalıştırma

### 1. Web Arayüzü (Önerilen)
Modern web arayüzünü ve interaktif kartal sahnesini başlatmak için:

```bash
uv run tdrag-web
# Veya: uv run python -m tdrag.web_api
```
Tarayıcınızdan **`http://localhost:8000`** adresine gidin.

### 2. Terminal CLI Modu
Terminal üzerinden hızlıca soru sormak için:

```bash
uv run tdrag
```

---

## ⚙️ Yapılandırma (`.env`)

Tüm parametreler `.env` dosyası üzerinden ayarlanabilir:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `TDRAG_LLM_PROVIDER` | `ollama` | LLM sağlayıcısı (`ollama`, `mock`) |
| `TDRAG_OLLAMA_MODEL` | `qwen3.5:4b` | Kullanılan yerel dil modeli |
| `TDRAG_LLM_NUM_CTX` | `6144` | Bağlam penceresi boyutu (6K) |
| `TDRAG_LLM_MAX_OUTPUT_TOKENS` | `500` | Modelin üretebileceği maksimum yanıt uzunluğu |
| `TDRAG_EMBEDDING_PROVIDER` | `sentence_transformers` | Vektörleme modeli (`paraphrase-multilingual-MiniLM-L12-v2`) |
| `TDRAG_TOP_K` | `4` | Sorgu başına getirilecek en alakalı parça sayısı |
| `TDRAG_SIMILARITY_THRESHOLD` | `0.25` | Alaka düzeyi alt sınırı (altındakiler elenir) |
| `TDRAG_MAX_VERIFICATION_ATTEMPTS` | `3` | Doğrulama döngüsü maksimum tekrar sayısı |
| `TDRAG_ARTICLES_FOLDER` | `./makaleler` | PDF dosyalarının taranacağı dizin |

---

## 🧪 Testler

Sistem, internete veya harici API'lere bağımlı olmadan izole bir şekilde 64 birim testten geçmektedir:

```bash
uv run pytest -v
```

Test kapsamı:
- **`test_chunker`**: Kelime sınırları, örtüşme pencereleri ve uzun metin bölümleri.
- **`test_pdf_loader`**: Boş sayfa, dijital metin çıkarma ve hata toleransı.
- **`test_config`**: 6K bağlam doğrulaması, ortam değişkeni parse işlemleri.
- **`test_vector_store`**: Jaccard benzerlik elemesi, eşik filtreleme ve upsert mantığı.
- **`test_qa_pipeline`**: Üret-doğrula döngüsü, red durumları, bozuk JSON onarımı ve asgari uzunluk denetimi.
- **`test_web_api`**: FastAPI uç noktaları, statik dosya sunumu ve Historia Aquilae marka kontrolleri.

---

## 📂 Proje Yapısı

```text
turk_devletleri_rag/
├── makaleler/                      # Akademik PDF vesikaları arşivi
│   ├── 864443.pdf                  # Normanlar ve Sicilya Fethi makalesi
│   ├── Gokturk_Kaganligi.pdf
│   └── ...                         # Diğer Türk devletleri makaleleri
├── src/
│   └── tdrag/
│       ├── config.py               # Doğrulamalı Pydantic yapılandırması
│       ├── models.py               # Chunk, Answer, VerificationResult veri modelleri
│       ├── pdf_loader.py           # pdfplumber metin ayrıştırıcı
│       ├── chunker.py              # Kayan pencereli metin parçalayıcı
│       ├── embeddings.py           # Multilingual SentenceTransformer & Hashing
│       ├── vector_store.py         # ChromaDB sarmalayıcı (filtreli ve dedup destekli)
│       ├── prompts.py              # Sıfır halüsinasyon ve konu sapmasını önleyici promptlar
│       ├── qa_pipeline.py          # Üret -> Doğrula -> Kendi kendini düzelt döngüsü
│       ├── ingestion.py            # Otomatik ve idempotent indeksleyici
│       ├── cli.py                  # Terminal etkileşimli arayüzü
│       ├── web_api.py              # FastAPI Web & REST API sunucusu
│       ├── web/                    # "Historia Aquilae" Web Arayüzü
│       │   ├── index.html          # Semantik HTML5 & Cinzel tipografisi
│       │   ├── style.css           # The House of Da Vinci 3 temalı stil sistemi
│       │   ├── app.js              # İnteraktif kartal sahnesi, akordeon ve sohbet istemcisi
│       │   └── assets/             # İmparatorluk mührü ve kartal arka plan görselleri
│       └── providers/              # LLM sağlayıcıları (Ollama & Mock)
├── tests/                          # 64 adet kapsamlı birim testi
├── pyproject.toml                  # Proje bağımlılıkları ve script giriş noktaları
└── README.md                       # Proje dokümantasyonu
```

---

## 📜 Lisans & Atıf
Bu proje, akademik araştırmaları ve tarih vesikalarını şeffaf, doğrulanabilir ve sıfır halüsinasyon güvencesiyle dijital çağa taşımak amacıyla geliştirilmiştir.
