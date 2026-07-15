import pytest

from tdrag.config import Config


def test_varsayilan_config_hata_firlatmaz():
    Config()


@pytest.mark.parametrize("esik", [0.0, -0.1, 1.1, 2.0])
def test_gecersiz_similarity_threshold_reddedilir(esik):
    with pytest.raises(ValueError):
        Config(similarity_threshold=esik)


def test_overlap_size_esit_veya_buyukse_reddedilir():
    with pytest.raises(ValueError):
        Config(chunk_size_words=100, chunk_overlap_words=100)


def test_negatif_chunk_size_reddedilir():
    with pytest.raises(ValueError):
        Config(chunk_size_words=-10, chunk_overlap_words=-20)


def test_gecersiz_llm_provider_reddedilir():
    with pytest.raises(ValueError):
        Config(llm_provider="bilinmeyen_saglayici")


def test_gecersiz_embedding_backend_reddedilir():
    with pytest.raises(ValueError):
        Config(embedding_backend="bilinmeyen_backend")


def test_sifir_max_verification_attempts_reddedilir():
    with pytest.raises(ValueError):
        Config(max_verification_attempts=0)


def test_sifir_top_k_reddedilir():
    with pytest.raises(ValueError):
        Config(top_k=0)
