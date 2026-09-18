"""Klasördeki PDF'leri okuyup parçalayarak vektör veritabanına yazan akış.

Artımlı (incremental) yükleme: Daha önce indekslenmiş PDF'ler tekrar işlenmez,
böylece her çalıştırmada dakikalarca süren gereksiz PDF okuma ve embedding
hesaplama işlemi engellenir.
"""
from __future__ import annotations

import logging

from .chunker import chunk_text
from .config import Config
from .pdf_loader import load_articles
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_folder(config: Config, vector_store: VectorStore, force: bool = False) -> dict:
    """Klasördeki PDF'leri işler.
    
    Eğer dosyalar zaten veritabanında mevcutsa tekrar yükleme yapmaz.
    `force=True` verilirse tüm dosyalar baştan okunup upsert edilir.
    """
    if not config.articles_folder.is_dir():
        raise FileNotFoundError(f"Makale klasörü bulunamadı: {config.articles_folder}")

    mevcut_pdfler = {p.name for p in config.articles_folder.glob("*.pdf")}
    if not mevcut_pdfler:
        raise FileNotFoundError(f"{config.articles_folder} içinde hiç PDF bulunamadı.")

    zaten_indekslenenler = set() if force else vector_store.get_indexed_files()
    
    if not force and len(mevcut_pdfler - zaten_indekslenenler) == 0:
        return {
            "islenen_dosya_sayisi": 0,
            "zaten_var_olan_sayisi": len(zaten_indekslenenler),
            "atlanan_dosya_sayisi": 0,
            "atlanan_dosyalar": [],
            "toplam_parca_sayisi": vector_store.count(),
            "atlandi_mi": True,
        }

    islenmesi_gerekenler = (mevcut_pdfler - zaten_indekslenenler) if not force else None
    basarili, basarisiz = load_articles(config.articles_folder, only_files=islenmesi_gerekenler)

    for dosya_adi, hata in basarisiz:
        logger.warning("Atlandı - %s: %s", dosya_adi, hata)

    for dosya_adi, metin in basarili:
        parcalar = chunk_text(metin, dosya_adi, config.chunk_size_words, config.chunk_overlap_words)
        vector_store.add_chunks(parcalar)

    return {
        "islenen_dosya_sayisi": len(basarili),
        "zaten_var_olan_sayisi": len(zaten_indekslenenler),
        "atlanan_dosya_sayisi": len(basarisiz),
        "atlanan_dosyalar": [dosya_adi for dosya_adi, _ in basarisiz],
        "toplam_parca_sayisi": vector_store.count(),
        "atlandi_mi": False,
    }
