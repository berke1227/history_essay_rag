"""PDF -> düz metin çıkarma.

Bilinçli mimari sınır: OCR desteklenmez. Makaleler taranmış görüntü ise
(metin katmanı yoksa) bu modül hatayı açıkça bildirir ve durur; sessizce
boş/bozuk metinle devam etmez. Gerekirse OCR ayrı bir adım olarak eklenebilir.
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber

MIN_GECERLI_METIN_UZUNLUGU = 20


class PdfMetniCikarilamadiError(Exception):
    """PDF'ten yeterli metin çıkarılamadığında (muhtemelen taranmış görüntü)."""


def extract_text(pdf_path: Path) -> str:
    """Tek bir PDF'ten metni çıkarır. Metin katmanı yoksa/çok kısaysa hata fırlatır."""
    with pdfplumber.open(pdf_path) as pdf:
        sayfa_metinleri = [sayfa.extract_text() or "" for sayfa in pdf.pages]
    metin = "\n\n".join(sayfa_metinleri).strip()

    if len(metin) < MIN_GECERLI_METIN_UZUNLUGU:
        raise PdfMetniCikarilamadiError(
            f"{pdf_path.name}: metin katmanı bulunamadı veya çok kısa "
            "(muhtemelen taranmış görüntü içeriyor, OCR desteklenmiyor)."
        )
    return metin


def load_articles(
    folder: Path, only_files: set[str] | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Klasördeki PDF'leri okur. `only_files` verilirse yalnızca belirtilen dosyalar işlenir.

    Returns:
        (basarili, basarisiz) — basarili: [(dosya_adi, metin), ...],
        basarisiz: [(dosya_adi, hata_mesaji), ...]. Tek bir bozuk dosya
        tüm toplu işlemi durdurmaz.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Makale klasörü bulunamadı: {folder}")

    pdf_dosyalari = sorted(folder.glob("*.pdf"))
    if not pdf_dosyalari:
        raise FileNotFoundError(f"{folder} içinde hiç PDF bulunamadı.")

    if only_files is not None:
        pdf_dosyalari = [p for p in pdf_dosyalari if p.name in only_files]
        if not pdf_dosyalari:
            return [], []

    basarili: list[tuple[str, str]] = []
    basarisiz: list[tuple[str, str]] = []
    for pdf_path in pdf_dosyalari:
        try:
            metin = extract_text(pdf_path)
            basarili.append((pdf_path.name, metin))
        except (PdfMetniCikarilamadiError, Exception) as exc:  # noqa: BLE001 - toplu işlemde tek dosya hatası akışı durdurmamalı
            basarisiz.append((pdf_path.name, str(exc)))

    return basarili, basarisiz
