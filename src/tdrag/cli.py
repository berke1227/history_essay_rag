"""Komut satırı giriş noktası: içe aktarım + etkileşimli soru-cevap döngüsü."""
from __future__ import annotations

import logging

from .config import Config
from .embeddings import HashingEmbeddingFunction, SentenceTransformerEmbeddingFunction
from .ingestion import ingest_folder
from .providers import create_provider
from .qa_pipeline import answer_question
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _embedding_fonksiyonu_olustur(config: Config):
    if config.embedding_backend == "hashing":
        return HashingEmbeddingFunction()
    return SentenceTransformerEmbeddingFunction(config.embedding_model)


def main() -> None:
    config = Config()
    vector_store = VectorStore(
        persist_dir=config.chroma_persist_dir,
        collection_name=config.collection_name,
        embedding_function=_embedding_fonksiyonu_olustur(config),
        similarity_threshold=config.similarity_threshold,
        dedup_jaccard_threshold=config.dedup_jaccard_threshold,
    )

    print(f"'{config.articles_folder}' klasöründeki makaleler işleniyor...")
    sonuc = ingest_folder(config, vector_store)
    print(
        f"{sonuc['islenen_dosya_sayisi']} makale, {sonuc['toplam_parca_sayisi']} parça "
        f"olarak indekslendi. ({sonuc['atlanan_dosya_sayisi']} dosya atlandı)"
    )
    if sonuc["atlanan_dosyalar"]:
        print("Atlanan dosyalar:", ", ".join(sonuc["atlanan_dosyalar"]))

    llm = create_provider(config)
    print("\nSorularınızı yazın (çıkmak için 'q'):")

    while True:
        try:
            soru = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz.")
            break

        if soru.lower() in {"q", "quit", "exit"}:
            break
        if not soru:
            continue

        cevap = answer_question(soru, vector_store, llm, config)
        print(f"\n{cevap.text}")
        if cevap.sources:
            print(
                f"(Kaynaklar: {', '.join(cevap.sources)} | "
                f"deneme: {cevap.attempts} | grounded: {cevap.grounded})"
            )


if __name__ == "__main__":
    main()
