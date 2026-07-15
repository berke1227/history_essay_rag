"""Ortak pytest fixture'ları."""
from __future__ import annotations

from pathlib import Path

import pytest
from generate_sample_pdfs import olustur

from tdrag.config import Config
from tdrag.embeddings import HashingEmbeddingFunction
from tdrag.vector_store import VectorStore


@pytest.fixture
def sample_pdf_folder(tmp_path: Path) -> Path:
    """3 sentetik test makalesi içeren geçici bir klasör oluşturur."""
    klasor = tmp_path / "makaleler"
    olustur(klasor)
    return klasor


@pytest.fixture
def test_config(tmp_path: Path, sample_pdf_folder: Path) -> Config:
    """Ağa çıkmayan, izole (geçici klasörlerde çalışan) test konfigürasyonu.

    similarity_threshold düşük tutulur: HashingEmbeddingFunction kaba bir
    kelime-çakışma embedding'i olduğundan mutlak skorları gerçek bir
    sentence-transformer modelinden çok daha düşüktür (bkz. config.py'deki
    not). Eşik kalibrasyonu değil, boru hattı mekaniği test edilir.
    """
    return Config(
        articles_folder=sample_pdf_folder,
        chroma_persist_dir=tmp_path / ".chroma",
        embedding_backend="hashing",
        llm_provider="mock",
        similarity_threshold=0.05,
        top_k=3,
    )


@pytest.fixture
def test_vector_store(test_config: Config) -> VectorStore:
    return VectorStore(
        persist_dir=test_config.chroma_persist_dir,
        collection_name=test_config.collection_name,
        embedding_function=HashingEmbeddingFunction(),
        similarity_threshold=test_config.similarity_threshold,
        dedup_jaccard_threshold=test_config.dedup_jaccard_threshold,
    )
