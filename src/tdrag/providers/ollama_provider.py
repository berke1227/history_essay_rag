"""Yerel Ollama sağlayıcısı."""
from __future__ import annotations


class OllamaProvider:
    def __init__(self, model: str):
        self._model = model

    def generate(self, prompt: str) -> str:
        import ollama  # lazy import — yalnızca bu sağlayıcı seçilince gerekli

        yanit = ollama.generate(model=self._model, prompt=prompt)
        return yanit["response"]
