"""Klasördeki PDF'leri okuyup parçalayarak vektör veritabanına yazan akış."""
from __future__ import annotations

import logging

from .chunker import chunk_text
from .config import Config
from .pdf_loader import load_articles
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_folder(config: Config, vector_store: VectorStore) -> dict:
    """Klasördeki tüm PDF'leri işler. Tek bir dosyanın okunamaması diğerlerini etkilemez."""
    basarili, basarisiz = load_articles(config.articles_folder)

    for dosya_adi, hata in basarisiz:
        logger.warning("Atlandı - %s: %s", dosya_adi, hata)

    toplam_parca = 0
    for dosya_adi, metin in basarili:
        parcalar = chunk_text(metin, dosya_adi, config.chunk_size_words, config.chunk_overlap_words)
        toplam_parca += vector_store.add_chunks(parcalar)

    return {
        "islenen_dosya_sayisi": len(basarili),
        "atlanan_dosya_sayisi": len(basarisiz),
        "atlanan_dosyalar": [dosya_adi for dosya_adi, _ in basarisiz],
        "toplam_parca_sayisi": toplam_parca,
    }
