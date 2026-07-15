"""Anthropic API sağlayıcısı."""
from __future__ import annotations

import os


class AnthropicProvider:
    def __init__(self, model: str, api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY bulunamadı. .env dosyasına ekleyin veya "
                "ortam değişkeni olarak tanımlayın."
            )

    def generate(self, prompt: str) -> str:
        import anthropic  # lazy import — yalnızca bu sağlayıcı seçilince gerekli

        client = anthropic.Anthropic(api_key=self._api_key)
        yanit = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return yanit.content[0].text
