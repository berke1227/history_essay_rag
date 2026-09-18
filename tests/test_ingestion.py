from tdrag.config import Config
from tdrag.ingestion import ingest_folder
from tdrag.vector_store import VectorStore


def test_ingest_folder_tum_makaleleri_isler(test_config: Config, test_vector_store: VectorStore):
    sonuc = ingest_folder(test_config, test_vector_store)

    assert sonuc["islenen_dosya_sayisi"] == 3
    assert sonuc["atlanan_dosya_sayisi"] == 0
    assert sonuc["toplam_parca_sayisi"] >= 3


def test_ingest_folder_bozuk_dosyayi_atlar_digerlerini_isler(
    test_config: Config, test_vector_store: VectorStore, sample_pdf_folder
):
    (sample_pdf_folder / "bozuk.pdf").write_text("gecersiz pdf")

    sonuc = ingest_folder(test_config, test_vector_store)

    assert sonuc["islenen_dosya_sayisi"] == 3
    assert sonuc["atlanan_dosya_sayisi"] == 1
    assert sonuc["atlanan_dosyalar"] == ["bozuk.pdf"]


def test_ingest_folder_zaten_varsa_tekrar_islemez(test_config: Config, test_vector_store: VectorStore):
    sonuc1 = ingest_folder(test_config, test_vector_store)
    assert sonuc1["islenen_dosya_sayisi"] == 3
    assert sonuc1["atlandi_mi"] is False

    sonuc2 = ingest_folder(test_config, test_vector_store)
    assert sonuc2["islenen_dosya_sayisi"] == 0
    assert sonuc2["atlandi_mi"] is True
    assert sonuc2["toplam_parca_sayisi"] == sonuc1["toplam_parca_sayisi"]
