"""LLM'e gönderilen prompt şablonları. Kurallar promptun içine gömülüdür;
model çıktıyı bu kurallara göre üretir, kod tarafında ayrıca zorlanmaz
(uzunluk/JSON biçimi hariç — onlar qa_pipeline'da doğrulanır)."""

ANSWER_PROMPT = """Sen, yalnızca sağlanan makale alıntılarına dayanarak çalışan bir tarih araştırma asistanısın.

KURALLAR:
1. SADECE aşağıdaki "MAKALE ALINTILARI" bölümünde geçen bilgileri kullan. Dışarıdan bilgi ekleme, tahmin yürütme veya araştırma yapma.
2. Eğer sorunun cevabı alıntılarda yoksa, başka hiçbir şey eklemeden şunu söyle: "Bu konu verilen makalelerde ele alınmamış."
3. Cevap bulunabiliyorsa en az bir paragraf uzunluğunda, açıklayıcı ve akıcı olmalı.
{geri_bildirim_bolumu}
MAKALE ALINTILARI:
{context}

SORU: {question}

CEVAP:"""

GERI_BILDIRIM_SABLONU = """
ÖNCEKİ DENEMEN REDDEDİLDİ. Şu sorunları düzelterek yeniden yanıtla: {feedback}
"""

VERIFY_PROMPT = """Sen bir cevap denetleyicisisin. Aşağıdaki soruyu, kaynak alıntıları ve üretilen cevabı incele.

SORU: {question}

KAYNAK ALINTILAR:
{context}

ÜRETİLEN CEVAP:
{answer}

Şu üç kriteri kontrol et:
1. Cevap SADECE kaynak alıntılara dayanıyor mu (uydurma/dışarıdan bilgi yok mu)?
2. Cevap en az bir paragraf uzunluğunda, dolu ve açıklayıcı mı?
3. Eğer soru kaynak alıntılarda geçmiyorsa, cevap bunu doğru şekilde belirtiyor mu (araştırma yapmaya çalışmadan)?

SADECE aşağıdaki JSON formatında yanıt ver, başka hiçbir metin ekleme:
{{"gecti": true veya false, "geri_bildirim": "sorun varsa kısa ve somut açıklama, yoksa boş string"}}"""


def answer_prompt_olustur(question: str, context: str, feedback: str | None) -> str:
    geri_bildirim_bolumu = GERI_BILDIRIM_SABLONU.format(feedback=feedback) if feedback else ""
    return ANSWER_PROMPT.format(
        context=context, question=question, geri_bildirim_bolumu=geri_bildirim_bolumu
    )


def verify_prompt_olustur(question: str, context: str, answer: str) -> str:
    return VERIFY_PROMPT.format(question=question, context=context, answer=answer)
