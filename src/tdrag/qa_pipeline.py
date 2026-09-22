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
    question: str,
    vector_store: VectorStore,
    llm: LLMProvider,
    config: Config,
    source_file: str | None = None,
) -> Answer:
    ilgili_parcalar = vector_store.query(question, top_k=config.top_k, source_file=source_file)

    if not ilgili_parcalar:
        return Answer(text=KAPSAM_DISI_MESAJI, grounded=True, attempts=0, sources=[])

    context = _baglam_olustur(ilgili_parcalar, max_words=config.max_context_words)
    kaynaklar = sorted({p.chunk.source_file for p in ilgili_parcalar})
    return _uret_ve_dogrula(question, context, kaynaklar, llm, config)


def _baglam_olustur(parcalar: list[RetrievedChunk], max_words: int | None = None) -> str:
    if not parcalar:
        return ""
    if max_words is None:
        return "\n\n---\n\n".join(
            f"[Kaynak: {p.chunk.source_file}]\n{p.chunk.text}" for p in parcalar
        )

    secilen_metinler: list[str] = []
    toplam_kelime = 0
    for p in parcalar:
        parca_metni = f"[Kaynak: {p.chunk.source_file}]\n{p.chunk.text}"
        kelime_sayisi = len(parca_metni.split())
        # En az ilk parçayı ekle; sonrakiler bütçeyi aşıyorsa eklemeyi durdur
        if secilen_metinler and (toplam_kelime + kelime_sayisi > max_words):
            break
        secilen_metinler.append(parca_metni)
        toplam_kelime += kelime_sayisi

    return "\n\n---\n\n".join(secilen_metinler)


def _uret_ve_dogrula(
    question: str, context: str, kaynaklar: list[str], llm: LLMProvider, config: Config
) -> Answer:
    geri_bildirim: str | None = None
    cevap_metni = ""

    for deneme in range(1, config.max_verification_attempts + 1):
        cevap_metni = llm.generate(
            answer_prompt_olustur(question, context, geri_bildirim),
            max_tokens=config.llm_max_output_tokens,
            num_ctx=config.llm_num_ctx,
        )
        dogrulama = _dogrula(question, context, cevap_metni, llm, config)

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


def _dogrula(
    question: str,
    context: str,
    answer: str,
    llm: LLMProvider,
    config: Config | None = None,
) -> VerificationResult:
    max_tokens = config.llm_verify_max_tokens if config is not None else 150
    num_ctx = config.llm_num_ctx if config is not None else None
    ham_yanit = llm.generate(
        verify_prompt_olustur(question, context, answer),
        max_tokens=max_tokens,
        num_ctx=num_ctx,
    )
    try:
        veri = json.loads(_json_bloguna_indir(ham_yanit))
        return VerificationResult(
            passed=bool(veri.get("gecti", False)), feedback=str(veri.get("geri_bildirim", ""))
        )
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning("Doğrulayıcı geçerli JSON döndürmedi, güvenli tarafta kalınıyor: %s", exc)
        return VerificationResult(passed=False, feedback="Doğrulayıcı yanıtı ayrıştırılamadı.")



def _json_bloguna_indir(metin: str) -> str:
    """Model JSON'u ```json ... ``` bloğuna sarmışsa veya etrafına açıklama eklemişse JSON nesnesini ayıklar."""
    import re

    metin = metin.strip()
    # 1. ```json ... ``` veya ``` ... ``` bloklarını ara
    kod_blogu = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", metin, re.DOTALL)
    if kod_blogu:
        ayiklanan = kod_blogu.group(1).strip()
    else:
        # 2. Metin içinde süslü parantez bloğunu ara
        suslu_parantez = re.search(r"(\{.*\})", metin, re.DOTALL)
        if suslu_parantez:
            ayiklanan = suslu_parantez.group(1).strip()
        elif "{" in metin:
            # Açılmış ama kapanmamış/kesilmiş JSON bloğu
            ayiklanan = metin[metin.find("{") :].strip()
        else:
            if metin.startswith("```"):
                metin = metin.strip("`").removeprefix("json").strip()
            ayiklanan = metin

    # Kesilmiş JSON tamiri (kapanmamış tırnak veya süslü parantez)
    if ayiklanan.startswith("{") and not ayiklanan.endswith("}"):
        if ayiklanan.count('"') % 2 != 0:
            ayiklanan = ayiklanan + '"'
        ayiklanan = ayiklanan + "}"

    return ayiklanan



def _kapsam_disi_mi(metin: str) -> bool:
    return "ele alınmamış" in metin.lower()


def _yeterli_uzunlukta(metin: str, min_kelime: int) -> bool:
    if _kapsam_disi_mi(metin):
        return True
    return len(metin.split()) >= min_kelime
