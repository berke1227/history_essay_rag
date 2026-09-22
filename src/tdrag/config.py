"""Sistem genelinde kullanılan, doğrulamalı konfigürasyon.

Tüm ayarlanabilir sabitler ortam değişkeni ile geçilebilir (bkz. .env.example).
Bu, kaynak kodu değiştirmeden farklı embedding modelleri / korpuslar için
ayarları kalibre edebilmek içindir — özellikle `similarity_threshold` embedding
modeline göre çok değişir (bkz. aşağıdaki not).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DESTEKLENEN_SAGLAYICILAR = {"ollama", "mock"}
DESTEKLENEN_EMBEDDING_BACKENDLERI = {"sentence-transformer", "ollama", "hashing"}


def _env_int(anahtar: str, varsayilan: int) -> int:
    return int(os.getenv(anahtar, str(varsayilan)))


def _env_float(anahtar: str, varsayilan: float) -> float:
    return float(os.getenv(anahtar, str(varsayilan)))


@dataclass(frozen=True)
class Config:
    """Uygulama ayarları. Tüm sabitler burada doğrulanır; başka yerde
    büyülü sayı (magic number) kullanılmaz.

    ÖNEMLİ - similarity_threshold hakkında: Bu değer embedding modeline
    göre kalibre edilmelidir. Varsayılan (0.25), üretim için önerilen
    SentenceTransformerEmbeddingFunction (paraphrase-multilingual-MiniLM)
    için kabaca uygun bir başlangıç noktasıdır. Test amaçlı
    HashingEmbeddingFunction çok daha düşük mutlak skorlar üretir (bkz.
    tests/) — embedding_backend'i değiştirirseniz bu eşiği de yeniden
    kalibre edin, aksi halde her sorgu "kapsam dışı" dönebilir.
    """

    articles_folder: Path = field(
        default_factory=lambda: Path(os.getenv("TDRAG_ARTICLES_FOLDER", "makaleler"))
    )
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(os.getenv("TDRAG_CHROMA_DIR", ".chroma"))
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("TDRAG_COLLECTION_NAME", "turk_devletleri")
    )

    chunk_size_words: int = field(default_factory=lambda: _env_int("TDRAG_CHUNK_SIZE_WORDS", 220))
    chunk_overlap_words: int = field(
        default_factory=lambda: _env_int("TDRAG_CHUNK_OVERLAP_WORDS", 40)
    )

    top_k: int = field(default_factory=lambda: _env_int("TDRAG_TOP_K", 4))
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("TDRAG_SIMILARITY_THRESHOLD", 0.25)
    )
    dedup_jaccard_threshold: float = field(
        default_factory=lambda: _env_float("TDRAG_DEDUP_JACCARD_THRESHOLD", 0.85)
    )

    max_verification_attempts: int = field(
        default_factory=lambda: _env_int("TDRAG_MAX_VERIFICATION_ATTEMPTS", 3)
    )
    min_answer_word_count: int = field(
        default_factory=lambda: _env_int("TDRAG_MIN_ANSWER_WORD_COUNT", 40)
    )

    # --- Token ve Bağlam Penceresi Sınırları ---
    llm_num_ctx: int = field(default_factory=lambda: _env_int("TDRAG_LLM_NUM_CTX", 6144))
    llm_max_output_tokens: int = field(
        default_factory=lambda: _env_int("TDRAG_LLM_MAX_OUTPUT_TOKENS", 500)
    )
    llm_verify_max_tokens: int = field(
        default_factory=lambda: _env_int("TDRAG_LLM_VERIFY_MAX_TOKENS", 250)
    )
    max_context_words: int = field(
        default_factory=lambda: _env_int("TDRAG_MAX_CONTEXT_WORDS", 1000)
    )

    llm_provider: str = field(default_factory=lambda: os.getenv("TDRAG_LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: os.getenv("TDRAG_LLM_MODEL", "qwen3.5:4b"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "TDRAG_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("TDRAG_EMBEDDING_BACKEND", "sentence-transformer")
    )

    def __post_init__(self) -> None:
        self._dogrula_esik("similarity_threshold", self.similarity_threshold)
        self._dogrula_esik("dedup_jaccard_threshold", self.dedup_jaccard_threshold)

        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError(
                "chunk_overlap_words, chunk_size_words'ten küçük olmalı "
                f"(alınan: overlap={self.chunk_overlap_words}, size={self.chunk_size_words})"
            )
        if self.chunk_size_words <= 0:
            raise ValueError(f"chunk_size_words pozitif olmalı, alınan: {self.chunk_size_words}")
        if self.top_k <= 0:
            raise ValueError(f"top_k pozitif olmalı, alınan: {self.top_k}")
        if self.max_verification_attempts < 1:
            raise ValueError(
                f"max_verification_attempts en az 1 olmalı, alınan: {self.max_verification_attempts}"
            )
        if self.min_answer_word_count < 1:
            raise ValueError(
                f"min_answer_word_count pozitif olmalı, alınan: {self.min_answer_word_count}"
            )
        if self.llm_num_ctx <= 0:
            raise ValueError(f"llm_num_ctx pozitif olmalı, alınan: {self.llm_num_ctx}")
        if self.llm_max_output_tokens <= 0:
            raise ValueError(
                f"llm_max_output_tokens pozitif olmalı, alınan: {self.llm_max_output_tokens}"
            )
        if self.llm_verify_max_tokens <= 0:
            raise ValueError(
                f"llm_verify_max_tokens pozitif olmalı, alınan: {self.llm_verify_max_tokens}"
            )
        if self.max_context_words <= 0:
            raise ValueError(f"max_context_words pozitif olmalı, alınan: {self.max_context_words}")
        if self.llm_max_output_tokens >= self.llm_num_ctx:
            raise ValueError(
                f"llm_max_output_tokens ({self.llm_max_output_tokens}) "
                f"llm_num_ctx'ten ({self.llm_num_ctx}) küçük olmalı"
            )
        if self.llm_provider not in DESTEKLENEN_SAGLAYICILAR:
            raise ValueError(
                f"Desteklenmeyen llm_provider: {self.llm_provider!r}. "
                f"Geçerli seçenekler: {sorted(DESTEKLENEN_SAGLAYICILAR)}"
            )
        if self.embedding_backend not in DESTEKLENEN_EMBEDDING_BACKENDLERI:
            raise ValueError(
                f"Desteklenmeyen embedding_backend: {self.embedding_backend!r}. "
                f"Geçerli seçenekler: {sorted(DESTEKLENEN_EMBEDDING_BACKENDLERI)}"
            )

    @staticmethod
    def _dogrula_esik(ad: str, deger: float) -> None:
        if not (0.0 < deger <= 1.0):
            raise ValueError(f"{ad}, (0, 1] aralığında olmalı, alınan: {deger}")
