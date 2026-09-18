"""config.llm_provider değerine göre doğru sağlayıcı sınıfını örnekleyen fabrika."""
from __future__ import annotations

from ..config import Config
from .base import LLMProvider
from .mock_provider import MockLLMProvider
from .ollama_provider import OllamaProvider

_SAGLAYICI_SINIFLARI = {
    "ollama": OllamaProvider,
}

__all__ = ["create_provider", "LLMProvider", "OllamaProvider", "MockLLMProvider"]


def create_provider(config: Config) -> LLMProvider:
    if config.llm_provider == "mock":
        return MockLLMProvider()

    if not config.llm_model:
        raise ValueError(
            f"{config.llm_provider!r} sağlayıcısı için TDRAG_LLM_MODEL tanımlanmamış "
            "(.env dosyasına veya ortam değişkenlerine ekleyin)."
        )

    if config.llm_provider not in _SAGLAYICI_SINIFLARI:
        raise ValueError(f"Desteklenmeyen sağlayıcı: {config.llm_provider!r}")

    sinif = _SAGLAYICI_SINIFLARI[config.llm_provider]
    if config.llm_provider == "ollama":
        return sinif(
            model=config.llm_model,
            default_num_ctx=config.llm_num_ctx,
            default_max_tokens=config.llm_max_output_tokens,
        )
    return sinif(model=config.llm_model)

