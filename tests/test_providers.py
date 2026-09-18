from unittest.mock import MagicMock, patch
import pytest

from tdrag.config import Config
from tdrag.prompts import verify_prompt_olustur
from tdrag.providers import create_provider
from tdrag.providers.mock_provider import MockLLMProvider
from tdrag.providers.ollama_provider import OllamaProvider


def test_mock_saglayici_secilince_mock_dondurur():
    config = Config(llm_provider="mock")
    assert isinstance(create_provider(config), MockLLMProvider)


def test_ollama_saglayici_secilince_ollama_provider_dondurur():
    config = Config(llm_provider="ollama", llm_model="qwen3.5:4b")
    provider = create_provider(config)
    assert isinstance(provider, OllamaProvider)


def test_bos_model_adiyla_ollama_secilirse_acik_hata_verir():
    config = Config(llm_provider="ollama", llm_model="")
    with pytest.raises(ValueError, match="TDRAG_LLM_MODEL"):
        create_provider(config)


def test_mock_saglayici_dogrulama_promptuna_json_dondurur():
    saglayici = MockLLMProvider()
    prompt = verify_prompt_olustur("soru", "bağlam", "cevap")
    assert '"gecti"' in saglayici.generate(prompt)


def test_mock_saglayici_scripted_modda_sirayla_doner():
    saglayici = MockLLMProvider(responses=["a", "b"])
    assert saglayici.generate("x") == "a"
    assert saglayici.generate("x") == "b"
    assert saglayici.generate("x") == "b"  # liste tükenince son eleman tekrarlanır


def test_ollama_provider_thinking_etiketlerini_temizler():
    provider = OllamaProvider(model="qwen3.5:4b", default_num_ctx=6144, default_max_tokens=450)
    sahte_yanit = MagicMock()
    sahte_yanit.response = "<think>Burada modelin uzun reasoning düşüncesi var.</think>Doğrudan nihai cevap."

    with patch("ollama.generate", return_value=sahte_yanit) as mock_gen:
        sonuc = provider.generate("test prompt")
        assert sonuc == "Doğrudan nihai cevap."
        # think=False ve varsayılan options parametrelerinin aktarıldığını doğrula
        mock_gen.assert_called_once_with(
            model="qwen3.5:4b",
            prompt="test prompt",
            think=False,
            options={"num_ctx": 6144, "num_predict": 450},
        )


def test_ollama_provider_ozel_token_ve_ctx_parametrelerini_aktarir():
    provider = OllamaProvider(model="qwen3.5:4b")
    sahte_yanit = MagicMock()
    sahte_yanit.response = "Özel yanıt"

    with patch("ollama.generate", return_value=sahte_yanit) as mock_gen:
        sonuc = provider.generate(
            "test prompt", max_tokens=150, num_ctx=3072, temperature=0.0
        )
        assert sonuc == "Özel yanıt"
        mock_gen.assert_called_once_with(
            model="qwen3.5:4b",
            prompt="test prompt",
            think=False,
            options={"num_ctx": 3072, "num_predict": 150, "temperature": 0.0},
        )

