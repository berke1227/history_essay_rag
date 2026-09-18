"""ChromaDB üzerinde ince bir katman: ekleme, benzerlik-eşikli sorgu ve
mükerrer (duplicate) parça filtreleme.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk, RetrievedChunk


def _jaccard(a: str, b: str) -> float:
    """İki metin arasındaki kelime kümesi benzerliği (0-1). Örtüşen
    chunk'ların (aynı makalenin komşu penceresi) getirilen sonuçlarda
    neredeyse aynı içerikle iki kez görünmesini engellemek için kullanılır."""
    kelimeler_a, kelimeler_b = set(a.lower().split()), set(b.lower().split())
    if not kelimeler_a or not kelimeler_b:
        return 0.0
    kesisim = len(kelimeler_a & kelimeler_b)
    birlesim = len(kelimeler_a | kelimeler_b)
    return kesisim / birlesim


class VectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_function: Any,
        similarity_threshold: float,
        dedup_jaccard_threshold: float,
    ):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self._threshold = similarity_threshold
        self._dedup_threshold = dedup_jaccard_threshold

    def count(self) -> int:
        """Koleksiyondaki toplam parça sayısını döndürür."""
        return self._collection.count()

    def get_indexed_files(self) -> set[str]:
        """Koleksiyonda halihazırda indekslenmiş benzersiz dosya adlarını döndürür."""
        if self._collection.count() == 0:
            return set()
        metalar = self._collection.get(include=["metadatas"])["metadatas"]
        return {m["source_file"] for m in metalar if m and "source_file" in m}

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Parçaları ekler. `upsert` kullanılır: aynı dosya tekrar işlenirse
        (aynı chunk id) kayıt çoğalmaz, üzerine yazılır — bu, ayrı bir
        'zaten var mı' kontrolüne gerek bırakmadan mükerrer içerik
        oluşumunu engeller."""
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source_file": c.source_file, "chunk_index": c.chunk_index} for c in chunks],
        )
        return len(chunks)

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        aday_sayisi = max(top_k * 3, top_k)
        sonuc = self._collection.query(query_texts=[question], n_results=aday_sayisi)

        adaylar = self._sonucu_adaylara_donustur(sonuc)
        esigi_gecenler = [a for a in adaylar if a.score >= self._threshold]
        esigi_gecenler.sort(key=lambda a: a.score, reverse=True)

        return self._mukerrerleri_ele(esigi_gecenler)[:top_k]

    def _sonucu_adaylara_donustur(self, sonuc: dict) -> list[RetrievedChunk]:
        if not sonuc["ids"] or not sonuc["ids"][0]:
            return []

        adaylar = []
        for i, chunk_id in enumerate(sonuc["ids"][0]):
            metadata = sonuc["metadatas"][0][i]
            mesafe = sonuc["distances"][0][i]
            benzerlik = 1.0 - mesafe  # cosine uzayında: similarity = 1 - distance
            chunk = Chunk(
                id=chunk_id,
                text=sonuc["documents"][0][i],
                source_file=metadata["source_file"],
                chunk_index=metadata["chunk_index"],
            )
            adaylar.append(RetrievedChunk(chunk=chunk, score=benzerlik))
        return adaylar

    def _mukerrerleri_ele(self, adaylar: list[RetrievedChunk]) -> list[RetrievedChunk]:
        secilenler: list[RetrievedChunk] = []
        for aday in adaylar:
            benzer_var_mi = any(
                _jaccard(aday.chunk.text, secilen.chunk.text) >= self._dedup_threshold
                for secilen in secilenler
            )
            if not benzer_var_mi:
                secilenler.append(aday)
        return secilenler
