import logging

from tdrag.config import Config
from tdrag.models import Chunk
from tdrag.providers.mock_provider import MockLLMProvider
from tdrag.qa_pipeline import KAPSAM_DISI_MESAJI, answer_question
from tdrag.vector_store import VectorStore

_MALAZGIRT_METNI = (
    "Malazgirt Savaşı 1071 yılında Bizans ve Selçuklular arasında yapıldı, "
    "Sultan Alparslan komutasındaki ordu zafer kazandı"
)


def _tek_parcali_store(store: VectorStore) -> VectorStore:
    store.add_chunks([Chunk(id="a::0", text=_MALAZGIRT_METNI, source_file="a.pdf", chunk_index=0)])
    return store


def _uzun_gecerli_cevap() -> str:
    return "Malazgirt Savaşı hakkında: " + " ".join(["kelime"] * 40)


def test_ilgili_parca_yoksa_llme_hic_gitmeden_kapsam_disi_doner(
    test_config: Config, test_vector_store: VectorStore
):
    llm = MockLLMProvider(responses=["BU YANIT HİÇ KULLANILMAMALI"])

    cevap = answer_question("tamamen alakasız ve konu dışı bir soru", test_vector_store, llm, test_config)

    assert cevap.text == KAPSAM_DISI_MESAJI
    assert cevap.grounded is True
    assert cevap.attempts == 0


def test_ilk_denemede_gecerli_cevap_kabul_edilir(test_config: Config, test_vector_store: VectorStore):
    store = _tek_parcali_store(test_vector_store)
    uzun_cevap = _uzun_gecerli_cevap()
    llm = MockLLMProvider(responses=[uzun_cevap, '{"gecti": true, "geri_bildirim": ""}'])

    cevap = answer_question("Malazgirt Savaşı ne zaman oldu", store, llm, test_config)

    assert cevap.grounded is True
    assert cevap.attempts == 1
    assert cevap.text == uzun_cevap
    assert cevap.sources == ["a.pdf"]


def test_ilk_deneme_reddedilir_ikincide_kabul_edilir(test_config: Config, test_vector_store: VectorStore):
    store = _tek_parcali_store(test_vector_store)
    uzun_cevap = _uzun_gecerli_cevap()
    llm = MockLLMProvider(
        responses=[
            "çok kısa cevap",
            '{"gecti": false, "geri_bildirim": "çok kısa, genişlet"}',
            uzun_cevap,
            '{"gecti": true, "geri_bildirim": ""}',
        ]
    )

    cevap = answer_question("Malazgirt Savaşı ne zaman oldu", store, llm, test_config)

    assert cevap.grounded is True
    assert cevap.attempts == 2
    assert cevap.text == uzun_cevap


def test_dogrulayici_surekli_reddederse_max_denemede_grounded_false_doner(
    test_config: Config, test_vector_store: VectorStore
):
    store = _tek_parcali_store(test_vector_store)
    tekli_tur = ["kısa cevap", '{"gecti": false, "geri_bildirim": "yetersiz"}']
    llm = MockLLMProvider(responses=tekli_tur * test_config.max_verification_attempts)

    cevap = answer_question("Malazgirt Savaşı ne zaman oldu", store, llm, test_config)

    assert cevap.grounded is False
    assert cevap.attempts == test_config.max_verification_attempts
    assert cevap.text == "kısa cevap"  # son denemenin (başarısız da olsa) cevabı döner


def test_dogrulayici_bozuk_json_donerse_guvenli_tarafta_kalinir(
    test_config: Config, test_vector_store: VectorStore, caplog
):
    store = _tek_parcali_store(test_vector_store)
    uzun_cevap = _uzun_gecerli_cevap()
    tekli_tur = [uzun_cevap, "bu bir JSON değil, düz metin açıklama"]
    llm = MockLLMProvider(responses=tekli_tur * test_config.max_verification_attempts)

    with caplog.at_level(logging.WARNING):
        cevap = answer_question("Malazgirt Savaşı ne zaman oldu", store, llm, test_config)

    assert cevap.grounded is False
    assert "ayrıştırılamadı" in caplog.text


def test_gecti_true_ama_cok_kisa_cevap_kod_seviyesinde_yine_de_reddedilir(
    test_config: Config, test_vector_store: VectorStore
):
    """Doğrulayıcı 'gecti: true' dese bile, en az 1 paragraf gereksinimi
    kod seviyesinde de (defense in depth) ayrıca zorlanır."""
    store = _tek_parcali_store(test_vector_store)
    tekli_tur = ["Kısa cevap.", '{"gecti": true, "geri_bildirim": ""}']
    llm = MockLLMProvider(responses=tekli_tur * test_config.max_verification_attempts)

    cevap = answer_question("Malazgirt Savaşı ne zaman oldu", store, llm, test_config)

    assert cevap.grounded is False
    assert cevap.attempts == test_config.max_verification_attempts


def test_kapsam_disi_mesaji_kisa_olsa_da_kabul_edilir(test_config: Config, test_vector_store: VectorStore):
    """Model, ilgili parça getirilmiş ama soruyu asıl karşılamıyorsa
    kapsam-dışı mesajını üretebilir; bu mesaj kısa olsa da geçerli sayılmalı."""
    store = _tek_parcali_store(test_vector_store)
    llm = MockLLMProvider(responses=[KAPSAM_DISI_MESAJI, '{"gecti": true, "geri_bildirim": ""}'])

    cevap = answer_question("Malazgirt Savaşı ne zaman oldu", store, llm, test_config)

    assert cevap.grounded is True
    assert cevap.text == KAPSAM_DISI_MESAJI


def test_baglam_olustur_kelime_butcesini_asar_ise_kirpar():
    from tdrag.models import RetrievedChunk
    from tdrag.qa_pipeline import _baglam_olustur

    p1 = RetrievedChunk(
        chunk=Chunk(id="1", text="kelime " * 20, source_file="doc1.pdf", chunk_index=0),
        score=0.9,
    )
    p2 = RetrievedChunk(
        chunk=Chunk(id="2", text="kelime " * 20, source_file="doc2.pdf", chunk_index=1),
        score=0.8,
    )

    # Bütçe 25 kelime: ilk parça (20 kelime + etiket) sığar, ikinci parça bütçeyi aşar
    baglam = _baglam_olustur([p1, p2], max_words=25)
    assert "doc1.pdf" in baglam
    assert "doc2.pdf" not in baglam


def test_answer_question_llm_generate_cagrisina_token_sinirlari_aktarilir(
    test_config: Config, test_vector_store: VectorStore
):
    from unittest.mock import MagicMock

    store = _tek_parcali_store(test_vector_store)
    mock_llm = MagicMock()
    # İlk çağrı (cevap üretimi), ikinci çağrı (doğrulama)
    mock_llm.generate.side_effect = [
        _uzun_gecerli_cevap(),
        '{"gecti": true, "geri_bildirim": ""}',
    ]

    answer_question("Malazgirt Savaşı ne zaman oldu", store, mock_llm, test_config)

    assert mock_llm.generate.call_count == 2
    ilk_cagri_kwargs = mock_llm.generate.call_args_list[0].kwargs
    assert ilk_cagri_kwargs["max_tokens"] == test_config.llm_max_output_tokens
    assert ilk_cagri_kwargs["num_ctx"] == test_config.llm_num_ctx

    ikinci_cagri_kwargs = mock_llm.generate.call_args_list[1].kwargs
    assert ikinci_cagri_kwargs["max_tokens"] == test_config.llm_verify_max_tokens
    assert ikinci_cagri_kwargs["num_ctx"] == test_config.llm_num_ctx

