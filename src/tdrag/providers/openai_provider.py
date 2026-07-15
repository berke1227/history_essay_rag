"""OpenAI API sağlayıcısı (base_url değiştirilerek OpenAI-uyumlu diğer
endpoint'ler için de kullanılabilir)."""
from __future__ import annotations

import os


class OpenAIProvider:
    def __init__(self, model: str, api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = os.getenv("OPENAI_BASE_URL")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY bulunamadı. .env dosyasına ekleyin veya "
                "ortam değişkeni olarak tanımlayın."
            )

    def generate(self, prompt: str) -> str:
        from openai import OpenAI  # lazy import — yalnızca bu sağlayıcı seçilince gerekli

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        yanit = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return yanit.choices[0].message.content
