"""Tüm LLM sağlayıcılarının uyması gereken ortak arayüz."""
from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str: ...

