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


def test_varsayilan_token_ve_baglam_degerleri():
    cfg = Config()
    assert cfg.llm_num_ctx == 6144
    assert cfg.llm_max_output_tokens == 500
    assert cfg.llm_verify_max_tokens == 250
    assert cfg.max_context_words == 1000
    assert cfg.top_k == 4


def test_negatif_veya_sifir_num_ctx_reddedilir():
    with pytest.raises(ValueError, match="llm_num_ctx"):
        Config(llm_num_ctx=0)


def test_negatif_veya_sifir_max_output_tokens_reddedilir():
    with pytest.raises(ValueError, match="llm_max_output_tokens"):
        Config(llm_max_output_tokens=-1)


def test_max_output_tokens_num_ctx_ten_buyuk_veya_esitse_reddedilir():
    with pytest.raises(ValueError, match="küçük olmalı"):
        Config(llm_num_ctx=500, llm_max_output_tokens=500)


def test_negatif_veya_sifir_max_context_words_reddedilir():
    with pytest.raises(ValueError, match="max_context_words"):
        Config(max_context_words=0)

