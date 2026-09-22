from pathlib import Path

from tdrag.embeddings import HashingEmbeddingFunction
from tdrag.models import Chunk
from tdrag.vector_store import VectorStore, _jaccard


def test_jaccard_ayni_metinler_bir_dondurur():
    assert _jaccard("bir iki üç", "bir iki üç") == 1.0


def test_jaccard_alakasiz_metinler_sifir_dondurur():
    assert _jaccard("bir iki üç", "dört beş altı") == 0.0


def test_jaccard_kismi_ortusme_orani_dogru_hesaplar():
    # {a,b,c} vs {a,b,d} -> kesişim=2, birleşim=4 -> 0.5
    assert _jaccard("a b c", "a b d") == 0.5


def test_jaccard_bos_metin_sifir_dondurur():
    assert _jaccard("", "bir iki") == 0.0


def test_add_ve_query_temel_akis_calisir(test_vector_store: VectorStore):
    test_vector_store.add_chunks(
        [Chunk(id="a::0", text="Malazgirt Savaşı 1071 yılında yapıldı", source_file="a.pdf", chunk_index=0)]
    )
    sonuclar = test_vector_store.query("Malazgirt Savaşı ne zaman yapıldı", top_k=3)
    assert len(sonuclar) == 1
    assert sonuclar[0].chunk.source_file == "a.pdf"


def test_upsert_ayni_id_ile_tekrar_eklemek_coklamaz(test_vector_store: VectorStore):
    parca = Chunk(id="a::0", text="Bursa 1326 yılında fethedildi", source_file="a.pdf", chunk_index=0)
    test_vector_store.add_chunks([parca])
    test_vector_store.add_chunks([parca])  # aynı dosya tekrar işlenmiş gibi

    sonuclar = test_vector_store.query("Bursa fethi", top_k=10)
    assert len(sonuclar) == 1


def test_esik_altindaki_sonuclar_filtrelenir(tmp_path: Path):
    vs = VectorStore(
        persist_dir=tmp_path / ".chroma_esik_testi",
        collection_name="esik_testi",
        embedding_function=HashingEmbeddingFunction(),
        similarity_threshold=0.99,  # kaba hashing embedding ile pratikte ulaşılamaz
        dedup_jaccard_threshold=0.85,
    )
    vs.add_chunks(
        [Chunk(id="x::0", text="tamamen farklı bir konu hakkında cümle", source_file="x.pdf", chunk_index=0)]
    )
    assert vs.query("bambaşka bir soru", top_k=5) == []


def test_dedup_neredeyse_ayni_parcalari_birlestirir(test_vector_store: VectorStore):
    metin_1 = "Malazgirt Savaşı 1071 yılında Bizans ve Selçuklular arasında yapıldı Alparslan komutanıydı"
    metin_2 = "Malazgirt Savaşı 1071 yılında Bizans ve Selçuklular arasında yapıldı Alparslan ordunun komutanıydı"
    farkli_metin = "Bursa 1326 yılında Orhan Bey tarafından fethedildi ve başkent yapıldı"

    test_vector_store.add_chunks(
        [
            Chunk(id="a::0", text=metin_1, source_file="a.pdf", chunk_index=0),
            Chunk(id="a::1", text=metin_2, source_file="a.pdf", chunk_index=1),
            Chunk(id="b::0", text=farkli_metin, source_file="b.pdf", chunk_index=0),
        ]
    )

    sonuclar = test_vector_store.query("Malazgirt Savaşı Selçuklular Bizans Alparslan", top_k=5)
    malazgirt_sonuclari = [r for r in sonuclar if "Malazgirt" in r.chunk.text]
    assert len(malazgirt_sonuclari) == 1


def test_query_source_file_filtreler(test_vector_store: VectorStore):
    test_vector_store.add_chunks(
        [
            Chunk(
                id="doc1::0",
                text="Normanlar Sicilya adasını 1060 yılında işgale başladı",
                source_file="normanlar.pdf",
                chunk_index=0,
            ),
            Chunk(
                id="doc2::0",
                text="Harezmşahlar Devleti Orta Asya'da hüküm sürdü",
                source_file="harezm.pdf",
                chunk_index=0,
            ),
        ]
    )
    # Sadece normanlar.pdf içinde ara
    sonuclar = test_vector_store.query(
        "Sicilya işgali ve tarih", top_k=5, source_file="normanlar.pdf"
    )
    assert len(sonuclar) >= 1
    assert all(r.chunk.source_file == "normanlar.pdf" for r in sonuclar)

    # Sadece harezm.pdf içinde ara
    sonuclar_harezm = test_vector_store.query(
        "hüküm sürdü", top_k=5, source_file="harezm.pdf"
    )
    assert len(sonuclar_harezm) >= 1
    assert all(r.chunk.source_file == "harezm.pdf" for r in sonuclar_harezm)
