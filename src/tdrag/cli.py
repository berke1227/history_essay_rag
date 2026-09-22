"""Komut satırı giriş noktası: içe aktarım + etkileşimli soru-cevap döngüsü."""
from __future__ import annotations

import logging
import sys

from .config import Config
from .embeddings import (
    HashingEmbeddingFunction,
    OllamaEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)
from .ingestion import ingest_folder
from .providers import create_provider
from .qa_pipeline import answer_question
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _embedding_fonksiyonu_olustur(config: Config):
    if config.embedding_backend == "hashing":
        return HashingEmbeddingFunction()
    if config.embedding_backend == "ollama":
        return OllamaEmbeddingFunction(config.embedding_model or "nomic-embed-text")
    return SentenceTransformerEmbeddingFunction(config.embedding_model)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"web", "serve", "ui"}:
        import uvicorn
        print("TDRAG Web Arayüzü başlatılıyor: http://localhost:8000")
        uvicorn.run("tdrag.web_api:app", host="0.0.0.0", port=8000, reload=False)
        return

    config = Config()
    vector_store = VectorStore(

        persist_dir=config.chroma_persist_dir,
        collection_name=config.collection_name,
        embedding_function=_embedding_fonksiyonu_olustur(config),
        similarity_threshold=config.similarity_threshold,
        dedup_jaccard_threshold=config.dedup_jaccard_threshold,
    )

    print(f"'{config.articles_folder}' klasörü ve vektör veritabanı kontrol ediliyor...")
    sonuc = ingest_folder(config, vector_store)
    if sonuc.get("atlandi_mi"):
        print(
            f"Vektör veritabanı hazır: {sonuc['zaten_var_olan_sayisi']} makale "
            f"({sonuc['toplam_parca_sayisi']} parça) zaten yüklü. Yeniden indeksleme atlandı."
        )
    else:
        print(
            f"{sonuc['islenen_dosya_sayisi']} makale indekslendi. "
            f"Toplam {sonuc['toplam_parca_sayisi']} parça hazır."
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
