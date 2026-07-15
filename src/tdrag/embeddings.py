"""ChromaDB'nin EmbeddingFunction arayüzünü uygulayan iki embedding sınıfı."""
from __future__ import annotations

import hashlib

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """Model indirmeden çalışan, deterministik embedding.

    Yalnızca test/geliştirme amaçlıdır — kelime çakışmasına dayanır, gerçek
    anlamsal (semantic) benzerlik yakalamaz. Üretimde
    SentenceTransformerEmbeddingFunction kullanılmalıdır.
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


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """Üretim amaçlı çok dilli embedding (Türkçe dahil).

    Ağır bağımlılık (sentence-transformers + torch) yalnızca bu sınıf
    gerçekten örneklenince (lazy import) yüklenir; mock/test akışları bu
    paketi hiç kurmak zorunda kalmaz.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
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
