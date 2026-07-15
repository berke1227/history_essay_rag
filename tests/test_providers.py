import pytest

from tdrag.config import Config
from tdrag.prompts import verify_prompt_olustur
from tdrag.providers import create_provider
from tdrag.providers.anthropic_provider import AnthropicProvider
from tdrag.providers.mock_provider import MockLLMProvider


def test_mock_saglayici_secilince_mock_dondurur():
    config = Config(llm_provider="mock")
    assert isinstance(create_provider(config), MockLLMProvider)


def test_bos_model_adiyla_gercek_saglayici_secilirse_acik_hata_verir():
    config = Config(llm_provider="anthropic", llm_model="")
    with pytest.raises(ValueError, match="TDRAG_LLM_MODEL"):
        create_provider(config)


def test_api_anahtari_yoksa_anthropic_provider_acik_hata_verir(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(model="claude-sonnet-5")


def test_mock_saglayici_dogrulama_promptuna_json_dondurur():
    saglayici = MockLLMProvider()
    prompt = verify_prompt_olustur("soru", "bağlam", "cevap")
    assert '"gecti"' in saglayici.generate(prompt)


def test_mock_saglayici_scripted_modda_sirayla_doner():
    saglayici = MockLLMProvider(responses=["a", "b"])
    assert saglayici.generate("x") == "a"
    assert saglayici.generate("x") == "b"
    assert saglayici.generate("x") == "b"  # liste tükenince son eleman tekrarlanır
