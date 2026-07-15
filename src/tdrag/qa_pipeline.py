"""Soru-cevap ana akışı: getir -> üret -> doğrula döngüsü.

Belgelendirilmiş sapma: "en gerçekçi cevap alana kadar kendini yenileyecek"
gereksinimi, sınırsız bir döngü yerine `config.max_verification_attempts`
ile sınırlanmıştır. Sebep: sınırsız döngü, doğrulayıcı asla tatmin
olmazsa sonsuz maliyet/gecikmeye yol açar. Sınıra ulaşılırsa son üretilen
cevap `grounded=False` bayrağıyla döndürülür; sessizce başarılı gibi
gösterilmez.
"""
from __future__ import annotations

import json
import logging

from .config import Config
from .models import Answer, RetrievedChunk, VerificationResult
from .prompts import answer_prompt_olustur, verify_prompt_olustur
from .providers.base import LLMProvider
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

KAPSAM_DISI_MESAJI = "Bu konu verilen makalelerde ele alınmamış."


def answer_question(
    question: str, vector_store: VectorStore, llm: LLMProvider, config: Config
) -> Answer:
    ilgili_parcalar = vector_store.query(question, top_k=config.top_k)

    if not ilgili_parcalar:
        return Answer(text=KAPSAM_DISI_MESAJI, grounded=True, attempts=0, sources=[])

    context = _baglam_olustur(ilgili_parcalar)
    kaynaklar = sorted({p.chunk.source_file for p in ilgili_parcalar})
    return _uret_ve_dogrula(question, context, kaynaklar, llm, config)


def _baglam_olustur(parcalar: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(f"[Kaynak: {p.chunk.source_file}]\n{p.chunk.text}" for p in parcalar)


def _uret_ve_dogrula(
    question: str, context: str, kaynaklar: list[str], llm: LLMProvider, config: Config
) -> Answer:
    geri_bildirim: str | None = None
    cevap_metni = ""

    for deneme in range(1, config.max_verification_attempts + 1):
        cevap_metni = llm.generate(answer_prompt_olustur(question, context, geri_bildirim))
        dogrulama = _dogrula(question, context, cevap_metni, llm)

        if dogrulama.passed and _yeterli_uzunlukta(cevap_metni, config.min_answer_word_count):
            return Answer(text=cevap_metni, grounded=True, attempts=deneme, sources=kaynaklar)

        geri_bildirim = dogrulama.feedback or "Cevap yeterince uzun/somut değildi, genişlet."
        logger.warning(
            "Doğrulama başarısız (deneme %d/%d): %s",
            deneme,
            config.max_verification_attempts,
            geri_bildirim,
        )

    return Answer(
        text=cevap_metni, grounded=False, attempts=config.max_verification_attempts, sources=kaynaklar
    )


def _dogrula(question: str, context: str, answer: str, llm: LLMProvider) -> VerificationResult:
    ham_yanit = llm.generate(verify_prompt_olustur(question, context, answer))
    try:
        veri = json.loads(_json_bloguna_indir(ham_yanit))
        return VerificationResult(
            passed=bool(veri.get("gecti", False)), feedback=str(veri.get("geri_bildirim", ""))
        )
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning("Doğrulayıcı geçerli JSON döndürmedi, güvenli tarafta kalınıyor: %s", exc)
        return VerificationResult(passed=False, feedback="Doğrulayıcı yanıtı ayrıştırılamadı.")


def _json_bloguna_indir(metin: str) -> str:
    """Model JSON'u ```json ... ``` bloğuna sarmışsa temizler."""
    metin = metin.strip()
    if metin.startswith("```"):
        metin = metin.strip("`").removeprefix("json").strip()
    return metin


def _kapsam_disi_mi(metin: str) -> bool:
    return "ele alınmamış" in metin.lower()


def _yeterli_uzunlukta(metin: str, min_kelime: int) -> bool:
    if _kapsam_disi_mi(metin):
        return True
    return len(metin.split()) >= min_kelime
