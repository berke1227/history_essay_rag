"""Tüm LLM sağlayıcılarının uyması gereken ortak arayüz."""
from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
