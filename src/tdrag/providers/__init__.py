"""config.llm_provider değerine göre doğru sağlayıcı sınıfını örnekleyen fabrika."""
from __future__ import annotations

from ..config import Config
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .mock_provider import MockLLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

_SAGLAYICI_SINIFLARI = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}

__all__ = ["create_provider", "LLMProvider"]


def create_provider(config: Config) -> LLMProvider:
    if config.llm_provider == "mock":
        return MockLLMProvider()

    if not config.llm_model:
        raise ValueError(
            f"{config.llm_provider!r} sağlayıcısı için TDRAG_LLM_MODEL tanımlanmamış "
            "(.env dosyasına veya ortam değişkenlerine ekleyin)."
        )

    sinif = _SAGLAYICI_SINIFLARI[config.llm_provider]
    return sinif(model=config.llm_model)
