"""Yerel Ollama sağlayıcısı."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class OllamaProvider:
    def __init__(
        self,
        model: str,
        default_num_ctx: int = 6144,
        default_max_tokens: int = 450,
    ):
        self._model = model
        self._default_num_ctx = default_num_ctx
        self._default_max_tokens = default_max_tokens

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        import ollama  # lazy import — yalnızca bu sağlayıcı seçilince gerekli

        options: dict[str, Any] = {}
        actual_ctx = num_ctx if num_ctx is not None else self._default_num_ctx
        if actual_ctx:
            options["num_ctx"] = actual_ctx

        actual_tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        if actual_tokens:
            options["num_predict"] = actual_tokens

        if temperature is not None:
            options["temperature"] = temperature

        yanit = ollama.generate(
            model=self._model,
            prompt=prompt,
            think=False,
            options=options,
        )

        ham_metin = getattr(yanit, "response", None)
        if ham_metin is None:
            if isinstance(yanit, dict):
                ham_metin = yanit.get("response", "")
            else:
                ham_metin = str(yanit)

        prompt_tokens = getattr(yanit, "prompt_eval_count", None)
        eval_tokens = getattr(yanit, "eval_count", None)
        total_duration = getattr(yanit, "total_duration", None)
        duration_ms = (total_duration / 1e6) if total_duration else 0.0

        logger.debug(
            "Ollama çıkarımı: model=%s, in_tok=%s, out_tok=%s, ctx=%s, max_tok=%s, süre=%.1fms",
            self._model,
            prompt_tokens,
            eval_tokens,
            options.get("num_ctx"),
            options.get("num_predict"),
            duration_ms,
        )

        # Modelden dönen olası <think>...</think> reasoning bloklarını temizle
        temizlenmis = re.sub(r"<think>.*?</think>", "", ham_metin, flags=re.DOTALL).strip()
        return temizlenmis

