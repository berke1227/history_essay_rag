"""Metni vektör veritabanına yazılacak parçalara (chunk) böler.

Mimari karar (belgelendirilmiş sapma): Paragraf sınırlarını korumaya
çalışmak yerine kelime bazlı kayan pencere (sliding window) kullanılır.
Sebep: (1) kelime sınırında kestiği için hiçbir zaman kelimeyi ortadan
bölmez, (2) çok uzun tek paragraflarda (akademik makalelerde sık görülür)
özel durum kodu gerektirmez, (3) test edilmesi/davranışının tahmin
edilmesi çok daha kolaydır. `overlap_words` sayesinde bağlam sürekliliği
korunur.
"""
from __future__ import annotations

from .models import Chunk


def chunk_text(
    text: str, source_file: str, chunk_size_words: int, overlap_words: int
) -> list[Chunk]:
    kelimeler = text.split()
    if not kelimeler:
        return []

    adim = chunk_size_words - overlap_words  # Config bunun pozitif olduğunu garanti eder
    parcalar: list[Chunk] = []
    chunk_index = 0
    baslangic = 0
    while baslangic < len(kelimeler):
        bitis = baslangic + chunk_size_words
        parca_metni = " ".join(kelimeler[baslangic:bitis])
        parcalar.append(
            Chunk(
                id=f"{source_file}::{chunk_index}",
                text=parca_metni,
                source_file=source_file,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1
        baslangic += adim

    return parcalar
