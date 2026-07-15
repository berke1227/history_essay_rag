"""Google Gemini API sağlayıcısı."""
from __future__ import annotations

import os


class GeminiProvider:
    def __init__(self, model: str, api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY bulunamadı. .env dosyasına ekleyin veya "
                "ortam değişkeni olarak tanımlayın."
            )

    def generate(self, prompt: str) -> str:
        import google.generativeai as genai  # lazy import — yalnızca bu sağlayıcı seçilince gerekli

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model)
        yanit = model.generate_content(prompt)
        return yanit.text
