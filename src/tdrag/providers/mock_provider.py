"""Test/demo amaçlı, ağa çıkmayan sağlayıcı.

İki modda çalışır:
- `responses` verilirse: sırayla bu yanıtları döndürür (birim testlerinde
  yeniden-deneme mantığını hassas biçimde test etmek için kullanılır).
- `responses` verilmezse: prompt içeriğine bakarak sahte ama tutarlı bir
  yanıt üretir (doğrulama promptuna JSON, cevap promptuna düz metin) —
  böylece gerçek bir API anahtarı olmadan tüm boru hattı uçtan uca denenebilir.
"""
from __future__ import annotations

_DOGRULAMA_ISARETI = "SADECE aşağıdaki JSON formatında"
_VARSAYILAN_CEVAP = (
    "Bu bir mock (sahte) sağlayıcı yanıtıdır. Gerçek bir cevap almak için "
    "TDRAG_LLM_PROVIDER değerini ollama/anthropic/openai/gemini olarak ayarlayıp "
    "TDRAG_LLM_MODEL ve ilgili API anahtarını tanımlayın."
)


class MockLLMProvider:
    def __init__(self, responses: list[str] | None = None):
        self._responses = responses
        self._index = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        if self._responses is not None:
            yanit = self._responses[min(self._index, len(self._responses) - 1)]
            self._index += 1
            return yanit

        if _DOGRULAMA_ISARETI in prompt:
            return '{"gecti": true, "geri_bildirim": ""}'
        return _VARSAYILAN_CEVAP

