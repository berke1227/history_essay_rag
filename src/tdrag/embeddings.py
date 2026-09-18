"""ChromaDB'nin EmbeddingFunction arayüzünü uygulayan embedding sınıfları."""
from __future__ import annotations

import hashlib
import os

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

# Hugging Face telemetri ve gereksiz online isteklerini engelle
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """Model indirmeden çalışan, deterministik embedding.

    Yalnızca test/geliştirme amaçlıdır — kelime çakışmasına dayanır, gerçek
    anlamsal (semantic) benzerlik yakalamaz.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "hashing-embedding-test-only"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(dim=config["dim"])

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(metin) for metin in input]

    def _embed_one(self, metin: str) -> list[float]:
        vektor = [0.0] * self.dim
        for kelime in metin.lower().split():
            h = int(hashlib.md5(kelime.encode("utf-8")).hexdigest(), 16)
            vektor[h % self.dim] += 1.0
        norm = sum(v * v for v in vektor) ** 0.5 or 1.0
        return [v / norm for v in vektor]


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Yerel Ollama üzerinden çalışan, internete bağımlı olmayan embedding.
    
    Varsayılan olarak nomic-embed-text modelini kullanır. Hızlıdır ve
    harici internet istekleri atmaz.
    """

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    @staticmethod
    def name() -> str:
        return "ollama-embedding"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(model_name=config["model_name"])

    def __call__(self, input: Documents) -> Embeddings:
        import ollama

        metinler = list(input)
        if not metinler:
            return []
        
        yanit = ollama.embed(model=self.model_name, input=metinler)
        return yanit.embeddings


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """Çok dilli sentence-transformer embedding sınıfı.
    
    Model diskte önbelleklenmişse internete istek atmadan yerel dosyaları
    kullanır (local_files_only), böylece gereksiz ağ hataları önlenir.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        try:
            # Öncelikli olarak yereldeki önbelleği kullan (ağ hatası vermemesi için)
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)

    @staticmethod
    def name() -> str:
        return "sentence-transformer"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "SentenceTransformerEmbeddingFunction":
        return SentenceTransformerEmbeddingFunction(model_name=config["model_name"])

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), normalize_embeddings=True).tolist()
